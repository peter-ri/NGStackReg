/*
    NGStackReg a fast image stack registration software
    Copyright (C) 2025  Peter D. Ringel

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

package ch.unibas.biozentrum.imagejplugins;

import com.jogamp.opencl.CLPlatform;

import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.app.StatusService;
import ch.unibas.biozentrum.imagejplugins.abstracts.RegistrationAndTransformation;
import ch.unibas.biozentrum.imagejplugins.implementations.HybridPrecisionNGStackReg;
import ch.unibas.biozentrum.imagejplugins.implementations.MTNGStackReg;
import ch.unibas.biozentrum.imagejplugins.implementations.OCLNGStackReg;
import ch.unibas.biozentrum.imagejplugins.implementations.SharedContext;
import ch.unibas.biozentrum.imagejplugins.implementations.SharedContextZT;
import ij.ImagePlus;
import ch.unibas.biozentrum.imagejplugins.implementations.AbstractSharedContext;
import com.jogamp.opencl.CLDevice;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import net.imagej.Dataset;
import net.imagej.Extents;
import net.imagej.ImgPlus;
import net.imagej.Position;
import net.imagej.axis.DefaultLinearAxis;
import net.imagej.display.DataView;
import net.imglib2.img.cell.CellImg;
import net.imglib2.img.planar.PlanarImg;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.ShortType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.type.numeric.integer.UnsignedLongType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import org.apache.commons.lang3.SystemUtils;

//TODO: not tested in headless mode

/**
 * @author Peter D. Ringel
 * @version 1.0.0
 *
 */

@Plugin(type = Command.class, headless = true, menuPath="Plugins>Registration>NGStackReg")
public class NGStackReg implements Command
{
    public static final int MIN_SIZE = 24;
    public enum TransformationType
    {
        INVALID,
        TRANSLATION,
        RIGIDBODY,
        SCALEDROTATION,
        AFFINE
    }
    private enum AlignmentAxisType
    {
        INVALID,
        CHANNELS,
        Z,
        TIME,
        ZANDT
    }
    
    @Parameter
    private LogService logService;
    @Parameter
    private StatusService statusService;
    @Parameter(label="Transformation:", choices={"Translation", "Rigid Body"/*,"Scaled Rotation","Affine"*/}) //The remaining options are not implemented yet
    private String sTransformationType;
    @Parameter(label="Alignment axis:", choices={"C","Z","T","Z -> T"})
    private String sAlignmentAxis;
    @Parameter(label="Alignment mode:", choices= {"GPU + CPU (hybrid prec.)", "GPU (hybrid prec.)", "CPU (hybrid prec.)", "CPU (double prec.)", "GPU (single prec.)", "GPU + CPU (double prec.)", "GPU (double prec.)"})
    private String alignmentMode;
    @Parameter(label="Save transformations to:", required = false)
    private File transformationOutput;
    @Parameter(persist = false)
    private Dataset dataset;
    @Parameter(persist = false, required = false)
    private DataView dview;
    @Parameter(persist = false, required = false)
    private ImagePlus imp;
    
    private boolean useFloatGPUOnly;
    private boolean useCPUOnly;
    private boolean doNotUseCPU;
    private boolean gpuAvailable;
    private TransformationType transformationType = TransformationType.INVALID;
    private AlignmentAxisType alignmentAxis = AlignmentAxisType.INVALID;
    @SuppressWarnings("rawtypes")
	private ImgPlus img;
    private RegistrationAndTransformation gpuimplementation;
    private RegistrationAndTransformation cpuimplementation;
    private int axisIndex = -1;
    private Position currentPos;
    private AbstractSharedContext sharedContext;
    
    private boolean forceDoublePrecisionRepr = false;
    
    /**
     * This main function serves only development purposes.
     * It allows to run the plugin immediately out of
     * the (IDE).
     *
     * @param args passed to fiji
     * @throws Exception
     */
    public static void main(final String... args)
    {
        sc.fiji.Main.main(args);
    }
    
    @Override
    public void run() {
        
        //First cleanup a bit
        Runtime.getRuntime().gc();
        statusService.clearStatus();
        
        //Check if a compatible OpenCL implementation and hardware are available
        gpuAvailable = hasGPUOpenCL();
        //If not translate the alignment mode to a CPU based alignment mode
        if(!gpuAvailable)
        {
	        switch(alignmentMode)
	        {
	            case "GPU + CPU (hybrid prec.)":
	            	alignmentMode = "CPU";
	                break;
	            case "GPU (hybrid prec.)":
	            	alignmentMode = "CPU";
	                break;
	            case "CPU (hybrid prec.)":
	                break;
	            case "CPU (double prec.)":
	                break;
	            case "GPU (single prec.)":
	            	alignmentMode = "CPU";
	                break;
	            case "GPU + CPU (double prec.)":
	            	alignmentMode = "CPU";
	            	forceDoublePrecisionRepr = true;
	            	break;
	            case "GPU (double prec.)":
	            	alignmentMode = "CPU";
	            	forceDoublePrecisionRepr = true;
	            	break;
	            default:
	                logService.error("The alignment mode is not supported");
	                return;
	        }
        }
        
        switch(alignmentMode)
        {
            case "GPU + CPU (hybrid prec.)":
            	useFloatGPUOnly = false;
            	useCPUOnly = false;
            	doNotUseCPU = false;
                break;
            case "GPU (hybrid prec.)":
            	useFloatGPUOnly = false;
            	useCPUOnly = false;
            	doNotUseCPU = true;
                break;
            case "CPU (hybrid prec.)":
            	useFloatGPUOnly = false;
            	useCPUOnly = true;
            	doNotUseCPU = false;
                break;
            case "CPU (double prec.)":
            	useFloatGPUOnly = false;
            	useCPUOnly = true;
            	doNotUseCPU = false;
            	forceDoublePrecisionRepr = true;
                break;
            case "GPU (single prec.)":
            	useFloatGPUOnly = true;
            	useCPUOnly = false;
            	doNotUseCPU = true;
                break;
            case "GPU + CPU (double prec.)":
            	useFloatGPUOnly = false;
            	useCPUOnly = false;
            	doNotUseCPU = false;
            	forceDoublePrecisionRepr = true;
            	break;
            case "GPU (double prec.)":
            	useFloatGPUOnly = false;
            	useCPUOnly = false;
            	doNotUseCPU = true;
            	forceDoublePrecisionRepr = true;
            	break;
            default:
                logService.error("The alignment mode is not supported");
                return;
        }
        
        //Now populate the information structures to start the job
        img = dataset.getImgPlus();
        if((!(img.getImg() instanceof PlanarImg)) && (!(img.getImg() instanceof CellImg)))
        {
            logService.warn("Image type is not supported. Only PlanarImg and CellImg are supoorted currently.");
            return;
        }
        if(dview != null)
        {
            //if the image is not shown ImageJ2 doesn't give it a dview
            currentPos = dview.getPlanePosition(); //known bug, this does not work when using the compatibility layer
            if(imp != null)
            {
            	if((img.numDimensions() - 2) <= 3)
            	{
            		long c = imp.getC() - 1; //Base 1 indexing
                	long z = imp.getZ() - 1;
                	long t = imp.getT() - 1;
                	int dims = img.numDimensions();
            		for(int i = 2;i < dims;i++)
                    {
            			if(!(img.axis(i) instanceof DefaultLinearAxis))
                        {
                            logService.error("The axis type is not supported.");
                            return;
                        }
            			if("Channel".equals(((DefaultLinearAxis)img.axis(i)).type().getLabel()))
                        {
            				currentPos.setPosition(c, i - 2); //Zero based index!
                        }
                        else if("Z".equals(((DefaultLinearAxis)img.axis(i)).type().getLabel()))
                        {
                        	currentPos.setPosition(z, i - 2); //Zero based index!
                        }
                        else if("Time".equals(((DefaultLinearAxis)img.axis(i)).type().getLabel()))
                        {
                        	currentPos.setPosition(t, i - 2); //Zero based index!
                        }
                    }
            	}
            }
        }
        else
        {
            long[] dims = new long[img.numDimensions() - 2];
            for(int i = 2;i < img.numDimensions();i++)
            {
                dims[i-2] = img.dimension(i);
            }
            Extents ext = new Extents(dims);
            currentPos = ext.createPosition(); //zero based index pointing to origin
        }
        //Position doesn't count X and Y as dimensions
        if(currentPos.numDimensions()+2 != img.numDimensions())
        {
            logService.error(currentPos);
            logService.error("The current positions dimensions " + currentPos.numDimensions() + " do not match the image dimensions " + img.numDimensions());
            return;
        }
        Extents extnts = currentPos.getExtents();
        for(int i = 0;i < currentPos.numDimensions();i++)
        {
            if(currentPos.getLongPosition(i) >= img.dimension(i+2) || currentPos.getLongPosition(i) < 0 || extnts.min(i) > 0)
            {
                logService.error("Illegal position or axis min > 0.");
                return;
            }
        }
        //Check that the image has a supported data type
        if((!(img.firstElement() instanceof ByteType)) && //8 bit signed
           (!(img.firstElement() instanceof UnsignedByteType)) && //8 bit unsigned
           (!(img.firstElement() instanceof ShortType)) && //16 bit signed
           (!(img.firstElement() instanceof UnsignedShortType)) && //16 bit unsigned
           (!(img.firstElement() instanceof IntType)) &&  //32 bit signed
           (!(img.firstElement() instanceof UnsignedIntType)) &&  //32 bit unsigned
           (!(img.firstElement() instanceof LongType)) && //64 bit signed
           (!(img.firstElement() instanceof UnsignedLongType)) && //64 bit unsigned
           (!(img.firstElement() instanceof FloatType)) && //float
           (!(img.firstElement() instanceof DoubleType))) //double
        {
            logService.error("The image datatype is not supported. Must be one of 8/16/32/64 bit signed/unsigned or float/double.");
            return;
        }
        //If these large data types are used the B-splines MUST be represented by double types, otherwise there will be a significant loss of precision
        if((img.firstElement() instanceof LongType) || (img.firstElement() instanceof DoubleType))
        {
            forceDoublePrecisionRepr = true;
            if(useFloatGPUOnly)
            {
                logService.warn("The datatype of the image (double/long/ulong) is too large to use only float approximation. Will use double precision code.");
                useFloatGPUOnly = false;
            }
        }
        switch(sTransformationType)
        {
            case "Translation":
                transformationType = TransformationType.TRANSLATION;
                break;
            case "Rigid Body":
                transformationType = TransformationType.RIGIDBODY;
                break;
            case "Scaled Rotation":
                transformationType = TransformationType.SCALEDROTATION;
                logService.error("Sorry currently only RIGIDBODY is supported.");
                return;
            case "Affine":
                transformationType = TransformationType.AFFINE;
                logService.error("Sorry currently only RIGIDBODY is supported.");
                return;
            default:
                logService.error("No such transformation type is supported");
                return;
        }
        switch(sAlignmentAxis)
        {
            case "C":
                alignmentAxis = AlignmentAxisType.CHANNELS;
                break;
            case "Z":
                alignmentAxis = AlignmentAxisType.Z;
                break;
            case "T":
                alignmentAxis = AlignmentAxisType.TIME;
                break;
            case "Z -> T":
            	alignmentAxis = AlignmentAxisType.ZANDT;
            	break;
            default:
                logService.error("No such axis supported.");
                return;
        }
        //Check the image dimensions (must be 2D at least)
        int dims = img.numDimensions();
        if(dims < 3)
        {
            logService.error("The image must have at least three dimensions (X, Y and one of C, Z or T).");
            return;
        }
        for(int i = 0; i < dims;i++)
        {
            if(!(img.axis(i) instanceof DefaultLinearAxis))
            {
                logService.error("The axis type is not supported.");
                return;
            }
        }
        
        if(!((DefaultLinearAxis)img.axis(0)).type().isXY() || !((DefaultLinearAxis)img.axis(1)).type().isXY())
        {
            logService.error("No X/Y axes.");
            return;
        }
        //check that the image is not too small
        if(img.dimension(0) < MIN_SIZE || img.dimension(1) < MIN_SIZE)
        {
            logService.error("The image width and height must be at least " + MIN_SIZE);
            return;
        }
        //In ImageJ2 the dimensions have standard names. Lets check the axes against these
        boolean hasAlignmentAxis = false;
        boolean hasZ = false;
        boolean hasT = false;
        int taxis = -1;
        for(int i = 2;i < dims;i++)
        {
            if(alignmentAxis == AlignmentAxisType.CHANNELS)
            {
                if("Channel".equals(((DefaultLinearAxis)img.axis(i)).type().getLabel()))
                {
                    if(img.dimension(i)>1)
                    {
                        hasAlignmentAxis = true;
                        axisIndex = i;
                    }
                    break;
                }
            }
            else if(alignmentAxis == AlignmentAxisType.Z)
            {
                if("Z".equals(((DefaultLinearAxis)img.axis(i)).type().getLabel()))
                {
                    if(img.dimension(i)>1)
                    {
                        hasAlignmentAxis = true;
                        axisIndex = i;
                    }
                    break;
                }
            }
            else if(alignmentAxis == AlignmentAxisType.TIME)
            {
                if("Time".equals(((DefaultLinearAxis)img.axis(i)).type().getLabel()))
                {
                    if(img.dimension(i)>1)
                    {
                        hasAlignmentAxis = true;
                        axisIndex = i;
                    }
                    break;
                }
            }
            else if(alignmentAxis == AlignmentAxisType.ZANDT)
            {
            	//The test for >(long)Integer.MAX_VALUE is done later for the other options.
            	//only this option requires a test on both axes.
            	if("Z".equals(((DefaultLinearAxis)img.axis(i)).type().getLabel()))
                {
                    if(img.dimension(i)>1)
                    {
                        axisIndex = i;
                        hasZ = true;
                        if(img.dimension(i) > (long)Integer.MAX_VALUE)
                        {
                            //of course this could be done differently, but then the transformations would 
                            //need to be calculated synchronously and here we are trying to be fast instead
                            //of compatible with large datasets.
                            logService.error("Axis is too large to allocate the transformations");
                            return;
                        }
                    }
                }
            	if("Time".equals(((DefaultLinearAxis)img.axis(i)).type().getLabel()))
                {
                    if(img.dimension(i)>1)
                    {
                    	taxis = i;
                        hasT = true;
                        if(img.dimension(i) > (long)Integer.MAX_VALUE)
                        {
                            //of course this could be done differently, but then the transformations would 
                            //need to be calculated synchronously and here we are trying to be fast instead
                            //of compatible with large datasets.
                            logService.error("Axis is too large to allocate the transformations");
                            return;
                        }
                    }
                }
            	if(hasZ && hasT)
            	{
        			//Since we are going to hardcode the transformation order this MUST have the
        			//order (C ->) Z -> T
            		if((axisIndex + 1) != taxis)
            		{
            			logService.error("Axis oder is not Z -> T, cannot align.");
                        return;
            		}
            		hasAlignmentAxis = true;
            		break;
            	}
            }
        }
        if(!hasAlignmentAxis)
        {
            //Nothing to do only one image on this axis namely this one so we are done
            return;
        }
        else
        {
            //Don't forget the alignment axes are not counting x and y coords so -2 but only for the positions
            axisIndex -= 2;
            taxis -= 2;
        }
        
        //Preallocate the matrix references but not the matrices as the type may vary depending on the GPU type
        if(img.dimension(axisIndex+2) > (long)Integer.MAX_VALUE)
        {
            logService.error("Axis is too large to allocate the transformations");
            return;
        }
        try
        {
        	if(alignmentAxis != AlignmentAxisType.ZANDT)
        	{
        		sharedContext = new SharedContext(transformationType, img,currentPos,axisIndex,forceDoublePrecisionRepr,logService,statusService);        		
        	}
        	else
        	{
        		sharedContext = new SharedContextZT(transformationType, img,currentPos, axisIndex, taxis, forceDoublePrecisionRepr,logService,statusService);
        	}
            if(img.dimension(axisIndex+2) > 1)
            {
                statusService.showStatus(0, (int)(img.dimension(axisIndex+2)-1), "Aligning");
            }

            //FIXME: Enable OpenCL accelerated code for non Windows OS
            //Unfortunately my test show that there is some kind of race condition happening with NVidia graphics card drivers on Ubuntu.
            //When reading back the buffers and writing them to disk everything works fine, but otherwise there is an unpredictable generation of NaNs.
            //Since I cannot figure out why this happens, I have decided to remove the OpenCL accelerated implementation when not running on Windows.
            if(!SystemUtils.IS_OS_WINDOWS)
            {
                useCPUOnly = true;
                useFloatGPUOnly = false;
                doNotUseCPU = false;
            }

            //Do not change the order of the instantiations as they set dependent data in the shared context
            //Test if OpenCL is available
            if((!useCPUOnly) && gpuAvailable)
            {
                logService.info("OpenCL is available and will be used.");
                if(forceDoublePrecisionRepr || useFloatGPUOnly)
                {
                    //requires double precision at all levels or only float precision
                    gpuimplementation = new OCLNGStackReg(useFloatGPUOnly,sharedContext);
                }
                else
                {
                    //can use float for intermediates but use double for last step
                    gpuimplementation = new HybridPrecisionNGStackReg(useFloatGPUOnly,sharedContext);
                }
                gpuimplementation.register();
            }
            else
            {
                logService.info("OpenCL is not available.");
            }
            if((gpuimplementation == null) && (doNotUseCPU || useFloatGPUOnly))
            {
            	logService.error("The OpenCL registration instances could not be created but GPU only alignment was chosen. Aborting.");
                return;
            }
            if(((!useFloatGPUOnly) || (gpuimplementation == null)) && (!doNotUseCPU))
            {
                //On typical systems don't use the processors when running float only code on the GPU because this usually only slows things down (unless of course you have >>8 cores)
                if(Runtime.getRuntime().availableProcessors() > sharedContext.getNrOfGPUDevices())
                {
                    //The GPUs are very resource hungry and need feeding threads (usually you have # of CPUs > # of GPUs though so just reserve some in the implementation
                    cpuimplementation = new MTNGStackReg(sharedContext);//There always will be a CPU
                    cpuimplementation.register();
                    cpuimplementation.waitForFinish();
                }
            }
            if(gpuimplementation != null)
            {
                gpuimplementation.waitForFinish();
            }
        }
        catch (Exception ex) {
            logService.error(ex.getMessage());
            return;
        }
        finally
        {
            //MUST always be performed to release acquired resources from OpenCL
            if(gpuimplementation != null)
            {
                gpuimplementation.release();
            }
            
            if(cpuimplementation != null)
            {
                cpuimplementation.release();
            }
            
        }
        statusService.clearStatus();
        dataset.update();
        //Now save the transformations if necessary
        if (transformationOutput != null) {
            if(transformationOutput.exists())
            {
                if (transformationOutput.canWrite()) {
                    try {
                        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(transformationOutput)));
                        sharedContext.serializeTransformations(out);
                        out.close();
                    } catch (IOException ex) {
                        logService.error(ex);
                    }
                } else {
                    logService.error("Cannot open the specified file for writing the transformations.");
                }
            }
            else
            {
                try {
                    PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(transformationOutput)));
                    sharedContext.serializeTransformations(out);
                    out.close();
                } catch (IOException ex) {
                    logService.error(ex);
                }
            }
                
        }
        Runtime.getRuntime().gc();
    }
    private boolean hasGPUOpenCL()
    {
        // In case the required libraries are not available this will fail with an exception.
        // In order to allow CPU only mode anyway all exceptions are caught and the GPU code
        // will not be used.
        try
        {
            if(!CLPlatform.isAvailable())
            {
                return false;
            }
            CLPlatform[] platforms = CLPlatform.listCLPlatforms();
            for(CLPlatform p: platforms)
            {
                CLDevice[] devs = p.listCLDevices();
                for(CLDevice d: devs)
                {
                    if(d.getType() == CLDevice.Type.GPU)
                    {
                        if(useFloatGPUOnly == false || forceDoublePrecisionRepr == true)
                        {
                            if(d.isDoubleFPAvailable() == true)
                            {
                                return true;
                            }
                        }
                        else
                        {
                            return true;
                        }
                    }
                }
            }
        }
        catch (Exception e)
        {
            
        }
        catch (NoClassDefFoundError classNotDef) 
        {
        	
        }
        return false;
    }
}
