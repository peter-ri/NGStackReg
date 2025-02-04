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

package ch.unibas.biozentrum.imagejplugins.implementations;
import ch.unibas.biozentrum.imagejplugins.NGStackReg;
import ch.unibas.biozentrum.imagejplugins.abstracts.ImageConverter;
import ch.unibas.biozentrum.imagejplugins.abstracts.RegistrationAndTransformation;
import ch.unibas.biozentrum.imagejplugins.util.RigidBodyTransformation;
import ch.unibas.biozentrum.imagejplugins.util.TranslationTransformation;
import ch.unibas.biozentrum.imagejplugins.util.imageconverters.ByteImageConverter;
import ch.unibas.biozentrum.imagejplugins.util.imageconverters.FloatImageConverter;
import ch.unibas.biozentrum.imagejplugins.util.imageconverters.IntImageConverter;
import ch.unibas.biozentrum.imagejplugins.util.imageconverters.ShortImageConverter;
import ch.unibas.biozentrum.imagejplugins.util.imageconverters.UnsignedByteImageConverter;
import ch.unibas.biozentrum.imagejplugins.util.imageconverters.UnsignedIntImageConverter;
import ch.unibas.biozentrum.imagejplugins.util.imageconverters.UnsignedShortImageConverter;
import ch.unibas.biozentrum.imagejplugins.util.StaticUtility;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory;
import com.jogamp.opencl.CLPlatform;
import com.jogamp.opencl.CLProgram;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.BrokenBarrierException;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.ShortType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * @author Peter D. Ringel
 * @version 1.0.0
 *
 */
public class HybridPrecisionNGStackReg extends RegistrationAndTransformation
{
    // Just constants for accessing the compiled kernels in the program (compiled separately for each device)
    private static final int NR_OF_OPENCL_KERNELS = 34;
    private static final int KERNEL_CubicBSplinePrefilter2DXlp = 0;
    private static final int KERNEL_CubicBSplinePrefilter2DYlp = 1;
    private static final int KERNEL_BasicToCardinal2DXhp = 2;
    private static final int KERNEL_BasicToCardinal2DYhp = 3;
    private static final int KERNEL_CubicBSplinePrefilter2DDeg7premulhp = 4;
    private static final int KERNEL_CubicBSplinePrefilter2DXDeg7lp = 5;
    private static final int KERNEL_CubicBSplinePrefilter2DYDeg7lp = 6;
    private static final int KERNEL_BasicToCardinal2DXhpDeg7 = 7;
    private static final int KERNEL_BasicToCardinal2DYhpDeg7 = 8;
    private static final int KERNEL_reduceDual1DX = 9;
    private static final int KERNEL_reduceDual1DY = 10;
    private static final int KERNEL_antiSymmetricFirMirrorOffBounds1DX = 11;
    private static final int KERNEL_antiSymmetricFirMirrorOffBounds1DY = 12;
    private static final int KERNEL_TargetedCubicBSplinePrefilter2Dpremulhp = 13;
    private static final int KERNEL_rigidBodyError = 14;
    private static final int KERNEL_rigidBodyErrorWithGradAndHess = 15;
    private static final int KERNEL_sumInLocalMemory = 16;
    private static final int KERNEL_parallelGroupedSumReduction = 17;
    private static final int KERNEL_sumInLocalMemoryCombined = 18;
    private static final int KERNEL_rigidBodyErrorWithGradAndHessBrent = 19;
    
    
    // the double kernels
    private static final int KERNEL_ConvertDoubleToFloat = 20;
    private static final int KERNEL_dCubicBSplinePrefilter2Dpremulhp = 21;
    private static final int KERNEL_dTargetedCubicBSplinePrefilter2Dpremulhp = 22;
    private static final int KERNEL_dCubicBSplinePrefilter2DXhp = 23;
    private static final int KERNEL_dCubicBSplinePrefilter2DYhp = 24;
    private static final int KERNEL_dantiSymmetricFirMirrorOffBounds1DX = 25;
    private static final int KERNEL_dantiSymmetricFirMirrorOffBounds1DY = 26;
    private static final int KERNEL_drigidBodyError = 27;
    private static final int KERNEL_drigidBodyErrorWithGradAndHess = 28;
    private static final int KERNEL_dsumInLocalMemory = 29;
    private static final int KERNEL_dsumInLocalMemoryCombined = 30;
    private static final int KERNEL_dparallelGroupedSumReduction = 31;
    private static final int KERNEL_drigidBodyErrorWithGradAndHessBrent = 32;
    private static final int KERNEL_dtransformImageWithBsplineInterpolation = 33;
    
    
    
    private static final int KERNEL_ftranslationError = 14; //KERNEL_rigidBodyError
    private static final int KERNEL_ftranslationErrorWithGradAndHess = 15; //KERNEL_rigidBodyErrorWithGradAndHess
    private static final int KERNEL_ftranslationSumInLocalMemoryCombined = 18; //KERNEL_sumInLocalMemoryCombined
    private static final int KERNEL_ftranslationErrorWithGradAndHessBrent = 19; //KERNEL_rigidBodyErrorWithGradAndHessBrent
    
    
    private static final int KERNEL_dtranslationError = 27; //KERNEL_drigidBodyError
    private static final int KERNEL_dtranslationErrorWithGradAndHess = 28; //KERNEL_drigidBodyErrorWithGradAndHess
    private static final int KERNEL_dtranslationSumInLocalMemoryCombined = 30; //KERNEL_dsumInLocalMemoryCombined
    private static final int KERNEL_dtranslationErrorWithGradAndHessBrent = 32; //KERNEL_drigidBodyErrorWithGradAndHessBrent
    private static final int KERNEL_dtranslationtransformImageWithBsplineInterpolation = 33; //KERNEL_dtransformImageWithBsplineInterpolation
    
    private static final int maximumSumReductionBlockNr = 64;//a maximum of 64 kernel blocks with a variable width will be started
    private static final int blocksizeMultiplier = 4;//optimal multiple * blocksizeMultiplier = blockSizes if this is less than the maximum number of elements that can be accommodated
    // Because the acquired devices can be very heterogeneous the block sizes have to be kept separate (they should be a multiple of the optimal multiple size)
    private int pyramidDepth = 1;
    private final boolean permitsFloatGPU;
    private CLContext[] contexts;
    private CLDevice[] devices = null;
    private static final CLMemory.Mem[] GPURESIDENTRW = {CLMemory.Mem.READ_WRITE};
    private HybridPrecisionNGStackRegWorker[] workers;
    /*
    Use an internal class because this has access to its parents attributes
    without having to explicitly pass them along.
    */
    private class HybridPrecisionNGStackRegWorker implements Runnable
    {
        /*
        This is the helper class encapsulating the code executed per thread.
        It implements the whole registration and transform code using a
        work stealing approach because if there are different GPUs available
        on the system, they may not be equivalent in their processing speed,
        which would lead to waiting for the slowest one when splitting the work
        equally.
        */
        private Thread t = null;
        private double offsetx = 0.0;
        private double offsety = 0.0;
        private double angle = 0.0;
        private double[][] hessian;
        private double[][] pseudoHessian;
        private double[] gradient;
        private int iterationPower;
        
        
        private int blockSizesFPT;
        private int blockSizesRigidBodyParallel;
        private long optimalMultiples[];
        private int maximumElementsForLocalFPTsum;
        private int maximumElementsForLocalFPTcombinedSum;
        
        private int doubleblockSizesFPT;
        private int doubleblockSizesRigidBodyParallel;
        private int doublemaximumElementsForLocalFPTsum;
        private int doublemaximumElementsForLocalFPTcombinedSum;
        
        private final CLContext context;
        private final CLDevice device;
        private CLCommandQueue queue = null;
        private CLCommandQueue asyncQueue = null;
        private OCLSourcePyramidSlice[] sourcePyramid;
        private OCLTargetPyramidSlice[] targetPyramid;
        
        private OCLSourcePyramidSlice sourceDoubleSlice;
        private OCLTargetPyramidSlice targetDoubleSlice;
        private double[] sourceImageDoubleSlice;
        private double[] sourcexGradientDoubleSlice;
        private double[] sourceyGradientDoubleSlice;
        private double[] targetCoefficientDoubleSlice;
        private double[] CPUentryImageBuffer;
        
        @SuppressWarnings("rawtypes")
		private CLBuffer conversionEntryBuffer;
        @SuppressWarnings("rawtypes")
		private CLBuffer entryImageBuffer;
        @SuppressWarnings("rawtypes")
        private CLBuffer fullSizedGPUResidentHelperBuffer;
        @SuppressWarnings("rawtypes")
        private CLBuffer secondaryGPUResidentHelperBuffer;
        
        @SuppressWarnings("rawtypes")
        private CLBuffer doubleEntryImageBuffer;
        @SuppressWarnings("rawtypes")
        private CLBuffer doubleFullSizedGPUResidentHelperBuffer;

        @SuppressWarnings("rawtypes")
        private CLBuffer[] parallelSumReductionBuffers;
        //These buffers are not necessary but the decision was to squeeze out more performance while using more memory
        @SuppressWarnings("rawtypes")
        private CLBuffer gradient0;
        @SuppressWarnings("rawtypes")
        private CLBuffer gradient1;
        @SuppressWarnings("rawtypes")
        private CLBuffer gradient2;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian00;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian01;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian02;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian11;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian12;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian22;

        @SuppressWarnings("rawtypes")
        private CLBuffer maskBuffer;
        private CLProgram conversionProgram;
        private CLKernel conversionProgramKernel;
        private CLKernel deConversionProgramKernel;

        private CLProgram uniformBSplineTransformProgram;
        private CLKernel uniformBSplineTransformProgramKernels[];
        
        private boolean usesFloat = false;
        private ImageConverter converter;
        
        private final int[] xInterpolationIndices;
        private final int[] yInterpolationIndices;
        private final double[] xWeights;
        private final double[] yWeights;
        
        private final SharedContextAlignmentTarget scat = new SharedContextAlignmentTarget();
        HybridPrecisionNGStackRegWorker(final CLContext context, final CLDevice device) throws Exception
        {
            switch(sharedContext.transformationType) {
            case TRANSLATION:
                hessian = new double[][] { {0.0,0.0},{0.0,0.0} };
                pseudoHessian = new double[][] { {0.0,0.0},{0.0,0.0} };
                gradient = new double[] {0.0,0.0};
                break;
            case RIGIDBODY:
            case SCALEDROTATION:
            case AFFINE:
            default:
                hessian = new double[][] { {0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0} };
                pseudoHessian = new double[][] { {0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0} };
                gradient = new double[] {0.0,0.0,0.0};
            }
            
            this.xWeights = new double[]{0.0, 0.0, 0.0, 0.0};
            this.yWeights = new double[]{0.0, 0.0, 0.0, 0.0};
            this.xInterpolationIndices = new int[]{ 0,0,0,0 };
            this.yInterpolationIndices = new int[]{ 0,0,0,0 };
            this.context = context;
            this.device = device;
            enumerateOCLDevicesAndInitialize();
            allocateMemory();
            CompileAndSetupOpenCLKernerls();
        }
        
        private void enumerateOCLDevicesAndInitialize() throws Exception
        {
            /*
            Created command queues with in-order execution by not setting out-of-order
            execution. According to the docs this allows one to call multiple kernels
            after one another without cross synchronization
            */
            if(!device.isDoubleFPAvailable())
            {
                usesFloat = true;
            }
            queue = device.createCommandQueue();
            if(device.getQueueProperties().contains(CLCommandQueue.Mode.OUT_OF_ORDER_MODE))
            {
                //Make it async if possible
                asyncQueue = device.createCommandQueue(CLCommandQueue.Mode.OUT_OF_ORDER_MODE);
            }
            else
            {
                asyncQueue = device.createCommandQueue();
            }
        }
        
        private void allocateMemory()
        {
            //allocate the pyramid memory
            sourcePyramid = new OCLSourcePyramidSlice[pyramidDepth];
            targetPyramid = new OCLTargetPyramidSlice[pyramidDepth];
            parallelSumReductionBuffers = new CLBuffer[2];// 2 parallel reduction buffers are needed

            int width = (int)sharedContext.img.dimension(0);
            int height = (int)sharedContext.img.dimension(1);
            if(width*height > Integer.MAX_VALUE)
            {
                throw new RuntimeException("Cannot allocate more than " + Integer.MAX_VALUE);
            }
            //This just defines the size of the buffers but they may be used for floats as well, for the last double precision step they need to be double though
            if(!usesFloat)
            {
                
                switch(sharedContext.transformationType) {
                case TRANSLATION:
                    maskBuffer = context.createDoubleBuffer((int)(width*height), GPURESIDENTRW);
                    maskBuffer.getCLSize();
                    gradient0 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                    gradient0.getCLSize();
                    gradient1 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                    gradient1.getCLSize();
                    hessian00 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                    hessian00.getCLSize();
                    hessian01 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                    hessian01.getCLSize();
                    hessian11 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                    hessian11.getCLSize();
                    for(int l = 0;l < 2;l++)
                    {
                        parallelSumReductionBuffers[l] = context.createDoubleBuffer(maximumSumReductionBlockNr, GPURESIDENTRW);
                        parallelSumReductionBuffers[l].getCLSize();
                    }
                    break;
                case RIGIDBODY:
                    maskBuffer = context.createDoubleBuffer((int)(width*height), GPURESIDENTRW);
                    maskBuffer.getCLSize();
                    gradient0 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                    gradient0.getCLSize();
                    gradient1 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                    gradient1.getCLSize();
                    gradient2 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                    gradient2.getCLSize();
                    hessian00 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                    hessian00.getCLSize();
                    hessian01 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                    hessian01.getCLSize();
                    hessian02 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                    hessian02.getCLSize();
                    hessian11 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                    hessian11.getCLSize();
                    hessian12 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                    hessian12.getCLSize();
                    hessian22 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                    hessian22.getCLSize();
                    for(int l = 0;l < 2;l++)
                    {
                        parallelSumReductionBuffers[l] = context.createDoubleBuffer(maximumSumReductionBlockNr, GPURESIDENTRW);
                        parallelSumReductionBuffers[l].getCLSize();
                    }
                    break;
                case SCALEDROTATION:
                    break;
                case AFFINE:
                    break;
                }
            }
            else
            {
                switch(sharedContext.transformationType) {
                case TRANSLATION:
                    maskBuffer = context.createFloatBuffer((int)(width*height), GPURESIDENTRW);
                    maskBuffer.getCLSize();
                    gradient0 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                    gradient0.getCLSize();
                    gradient1 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                    gradient1.getCLSize();
                    hessian00 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                    hessian00.getCLSize();
                    hessian01 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                    hessian01.getCLSize();
                    hessian11 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                    hessian11.getCLSize();
                    for(int l = 0;l < 2;l++)
                    {
                        parallelSumReductionBuffers[l] = context.createFloatBuffer(maximumSumReductionBlockNr, GPURESIDENTRW);
                        parallelSumReductionBuffers[l].getCLSize();
                    }
                    break;
                case RIGIDBODY:
                    maskBuffer = context.createFloatBuffer((int)(width*height), GPURESIDENTRW);
                    maskBuffer.getCLSize();
                    gradient0 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                    gradient0.getCLSize();
                    gradient1 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                    gradient1.getCLSize();
                    gradient2 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                    gradient2.getCLSize();
                    hessian00 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                    hessian00.getCLSize();
                    hessian01 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                    hessian01.getCLSize();
                    hessian02 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                    hessian02.getCLSize();
                    hessian11 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                    hessian11.getCLSize();
                    hessian12 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                    hessian12.getCLSize();
                    hessian22 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                    hessian22.getCLSize();
                    for(int l = 0;l < 2;l++)
                    {
                        parallelSumReductionBuffers[l] = context.createFloatBuffer(maximumSumReductionBlockNr, GPURESIDENTRW);
                        parallelSumReductionBuffers[l].getCLSize();
                    }
                    break;
                case SCALEDROTATION:
                    break;
                case AFFINE:
                    break;
                }
            }
            if(!usesFloat)
            {
                sourceDoubleSlice = new OCLSourcePyramidSlice();
                targetDoubleSlice = new OCLTargetPyramidSlice();
                // Everything can be done on the GPU
                sourceDoubleSlice.width = width;
                sourceDoubleSlice.height = height;
                sourceDoubleSlice.Image = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                sourceDoubleSlice.Image.getCLSize();
                sourceDoubleSlice.xGradient = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                sourceDoubleSlice.xGradient.getCLSize();
                sourceDoubleSlice.yGradient = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                sourceDoubleSlice.yGradient.getCLSize();
                targetDoubleSlice.width = width;
                targetDoubleSlice.height = height;
                targetDoubleSlice.Coefficient = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                targetDoubleSlice.Coefficient.getCLSize();
                // these are the conversion intermediate buffers
                doubleEntryImageBuffer = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                doubleEntryImageBuffer.getCLSize();
                doubleFullSizedGPUResidentHelperBuffer = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                doubleFullSizedGPUResidentHelperBuffer.getCLSize();
            }
            else
            {
                // Has to be done by the CPU so allocate the necessary buffers
                sourceImageDoubleSlice = new double[width*height];
                sourcexGradientDoubleSlice = new double[width*height];
                sourceyGradientDoubleSlice = new double[width*height];
                targetCoefficientDoubleSlice = new double[width*height];
                CPUentryImageBuffer = new double[width*height];
            }
            for(int j = 0;j < pyramidDepth; j++)
            {
                sourcePyramid[j] = new OCLSourcePyramidSlice();
                targetPyramid[j] = new OCLTargetPyramidSlice();
                sourcePyramid[j].width = width;
                sourcePyramid[j].height = height;
                targetPyramid[j].width = width;
                targetPyramid[j].height = height;   
                sourcePyramid[j].Image = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                sourcePyramid[j].Image.getCLSize();
                sourcePyramid[j].xGradient = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                sourcePyramid[j].xGradient.getCLSize();
                sourcePyramid[j].yGradient= context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                sourcePyramid[j].yGradient.getCLSize();
                targetPyramid[j].Coefficient = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                targetPyramid[j].Coefficient.getCLSize();
                width /= 2;
                height /= 2;
            }

            if(!usesFloat)
            {
                // this means that the double GPU implementation is not available so this will stay CPU side
                if((sharedContext.img.firstElement() instanceof ByteType)||(sharedContext.img.firstElement() instanceof UnsignedByteType))
                {
                    conversionEntryBuffer = context.createByteBuffer((int) (sharedContext.img.dimension(0)*sharedContext.img.dimension(1)), GPURESIDENTRW);
                }
                else if((sharedContext.img.firstElement() instanceof ShortType)||(sharedContext.img.firstElement() instanceof UnsignedShortType))
                {
                    conversionEntryBuffer = context.createShortBuffer((int) (sharedContext.img.dimension(0)*sharedContext.img.dimension(1)), GPURESIDENTRW);
                }
                else if((sharedContext.img.firstElement() instanceof IntType)||(sharedContext.img.firstElement() instanceof UnsignedIntType))
                {
                    conversionEntryBuffer = context.createIntBuffer((int) (sharedContext.img.dimension(0)*sharedContext.img.dimension(1)), GPURESIDENTRW);
                }
                else if((sharedContext.img.firstElement() instanceof FloatType))
                {
                    conversionEntryBuffer = context.createFloatBuffer((int) (sharedContext.img.dimension(0)*sharedContext.img.dimension(1)), GPURESIDENTRW);
                }
                else
                {
                    // long and double are not allowed for this method
                    throw new RuntimeException("Illegal image type");
                }
                conversionEntryBuffer.getCLSize();
            }
            else
            {
                if((sharedContext.img.firstElement() instanceof ByteType))
                {
                    converter = new ByteImageConverter();
                }
                else if((sharedContext.img.firstElement() instanceof UnsignedByteType))
                {
                    converter = new UnsignedByteImageConverter();
                }
                else if((sharedContext.img.firstElement() instanceof ShortType))
                {
                    converter = new ShortImageConverter();
                }
                else if((sharedContext.img.firstElement() instanceof UnsignedShortType))
                {
                    converter = new UnsignedShortImageConverter();
                }
                else if((sharedContext.img.firstElement() instanceof IntType))
                {
                    converter = new IntImageConverter();
                }
                else if((sharedContext.img.firstElement() instanceof UnsignedIntType))
                {
                    converter = new UnsignedIntImageConverter();
                }
                else if((sharedContext.img.firstElement() instanceof FloatType))
                {
                    converter = new FloatImageConverter();
                }
                else
                {
                    // long and double are not allowed for this method
                    throw new RuntimeException("Illegal image type");
                }
            }

            entryImageBuffer = context.createFloatBuffer((int) (sharedContext.img.dimension(0)*sharedContext.img.dimension(1)), GPURESIDENTRW);
            entryImageBuffer.getCLSize();
            fullSizedGPUResidentHelperBuffer = context.createFloatBuffer((int) (sharedContext.img.dimension(0)*sharedContext.img.dimension(1)), GPURESIDENTRW);
            fullSizedGPUResidentHelperBuffer.getCLSize();
            secondaryGPUResidentHelperBuffer = context.createFloatBuffer(((((int)sharedContext.img.dimension(0)))*(((int)sharedContext.img.dimension(1)))), GPURESIDENTRW);
            secondaryGPUResidentHelperBuffer.getCLSize();
        }
        
        private void CompileAndSetupOpenCLKernerls() throws IOException
        {
            uniformBSplineTransformProgramKernels = new CLKernel[NR_OF_OPENCL_KERNELS];
            optimalMultiples = new long[NR_OF_OPENCL_KERNELS];
            if(usesFloat)
            {
                // only get the float part
                uniformBSplineTransformProgram = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/HybridPrecisionBSplineTransform.cl")).build(device);
            }
            else
            {
                // also compile the double part
                uniformBSplineTransformProgram = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/HybridPrecisionBSplineTransform.cl")).build("-D USE_DOUBLE",device);
                uniformBSplineTransformProgramKernels[KERNEL_ConvertDoubleToFloat] = uniformBSplineTransformProgram.createCLKernel("ConvertDoubleToFloat");
                uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2Dpremulhp] = uniformBSplineTransformProgram.createCLKernel("dCubicBSplinePrefilter2Dpremulhp");
                uniformBSplineTransformProgramKernels[KERNEL_dTargetedCubicBSplinePrefilter2Dpremulhp] = uniformBSplineTransformProgram.createCLKernel("dTargetedCubicBSplinePrefilter2Dpremulhp");
                uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DXhp] = uniformBSplineTransformProgram.createCLKernel("dCubicBSplinePrefilter2DXhp");
                uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DYhp] = uniformBSplineTransformProgram.createCLKernel("dCubicBSplinePrefilter2DYhp");
                uniformBSplineTransformProgramKernels[KERNEL_dantiSymmetricFirMirrorOffBounds1DX] = uniformBSplineTransformProgram.createCLKernel("dantiSymmetricFirMirrorOffBounds1DX");
                uniformBSplineTransformProgramKernels[KERNEL_dantiSymmetricFirMirrorOffBounds1DY] = uniformBSplineTransformProgram.createCLKernel("dantiSymmetricFirMirrorOffBounds1DY");
                uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory] = uniformBSplineTransformProgram.createCLKernel("dsumInLocalMemory");
                uniformBSplineTransformProgramKernels[KERNEL_dparallelGroupedSumReduction] = uniformBSplineTransformProgram.createCLKernel("dparallelGroupedSumReduction");
                
                switch(sharedContext.transformationType) {
                case TRANSLATION:
                    uniformBSplineTransformProgramKernels[KERNEL_dtranslationError] = uniformBSplineTransformProgram.createCLKernel("dtranslationError");
                    uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHess] = uniformBSplineTransformProgram.createCLKernel("dtranslationErrorWithGradAndHess");
                    uniformBSplineTransformProgramKernels[KERNEL_dtranslationSumInLocalMemoryCombined] = uniformBSplineTransformProgram.createCLKernel("dtranslationSumInLocalMemoryCombined");
                    uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHessBrent] = uniformBSplineTransformProgram.createCLKernel("dtranslationErrorWithGradAndHessBrent");
                    uniformBSplineTransformProgramKernels[KERNEL_dtranslationtransformImageWithBsplineInterpolation] = uniformBSplineTransformProgram.createCLKernel("dtranslationtransformImageWithBsplineInterpolation");
                    break;
                case RIGIDBODY:
                    uniformBSplineTransformProgramKernels[KERNEL_drigidBodyError] = uniformBSplineTransformProgram.createCLKernel("drigidBodyError");
                    uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHess] = uniformBSplineTransformProgram.createCLKernel("drigidBodyErrorWithGradAndHess");
                    uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemoryCombined] = uniformBSplineTransformProgram.createCLKernel("dsumInLocalMemoryCombined");
                    uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHessBrent] = uniformBSplineTransformProgram.createCLKernel("drigidBodyErrorWithGradAndHessBrent");
                    uniformBSplineTransformProgramKernels[KERNEL_dtransformImageWithBsplineInterpolation] = uniformBSplineTransformProgram.createCLKernel("dtransformImageWithBsplineInterpolation");
                    break;
                case SCALEDROTATION:
                    break;
                case AFFINE:
                    break;
                }
                
                
                optimalMultiples[KERNEL_ConvertDoubleToFloat] = uniformBSplineTransformProgramKernels[KERNEL_ConvertDoubleToFloat].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_dCubicBSplinePrefilter2Dpremulhp] = uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2Dpremulhp].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_dTargetedCubicBSplinePrefilter2Dpremulhp] = uniformBSplineTransformProgramKernels[KERNEL_dTargetedCubicBSplinePrefilter2Dpremulhp].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_dCubicBSplinePrefilter2DXhp] = uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DXhp].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_dCubicBSplinePrefilter2DYhp] = uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DYhp].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_dantiSymmetricFirMirrorOffBounds1DX] = uniformBSplineTransformProgramKernels[KERNEL_dantiSymmetricFirMirrorOffBounds1DX].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_dantiSymmetricFirMirrorOffBounds1DY] = uniformBSplineTransformProgramKernels[KERNEL_dantiSymmetricFirMirrorOffBounds1DY].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_dsumInLocalMemory] = uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_dparallelGroupedSumReduction] = uniformBSplineTransformProgramKernels[KERNEL_dparallelGroupedSumReduction].getPreferredWorkGroupSizeMultiple(device);

                switch(sharedContext.transformationType) {
                case TRANSLATION:
                	optimalMultiples[KERNEL_dtranslationError] = uniformBSplineTransformProgramKernels[KERNEL_dtranslationError].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_dtranslationErrorWithGradAndHess] = uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHess].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_dtranslationSumInLocalMemoryCombined] = uniformBSplineTransformProgramKernels[KERNEL_dtranslationSumInLocalMemoryCombined].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_dtranslationErrorWithGradAndHessBrent] = uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHessBrent].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_dtranslationtransformImageWithBsplineInterpolation] = uniformBSplineTransformProgramKernels[KERNEL_dtranslationtransformImageWithBsplineInterpolation].getPreferredWorkGroupSizeMultiple(device);
                    
                    doubleblockSizesRigidBodyParallel = (int)Math.min(Math.min(uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHessBrent].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHessBrent].getLocalMemorySize(device))/((8)*11/*11 buffers are needed*/)), blocksizeMultiplier*optimalMultiples[KERNEL_dtranslationErrorWithGradAndHessBrent]);
                    doubleblockSizesRigidBodyParallel -= (doubleblockSizesRigidBodyParallel % optimalMultiples[KERNEL_dtranslationErrorWithGradAndHessBrent]);
                    if(doubleblockSizesRigidBodyParallel == 0)
                    {
                        doubleblockSizesRigidBodyParallel = 1;// Fallback solution minimum is 1
                    }
                    
                    doublemaximumElementsForLocalFPTcombinedSum = (int) Math.min(uniformBSplineTransformProgramKernels[KERNEL_dtranslationSumInLocalMemoryCombined].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_dtranslationSumInLocalMemoryCombined].getLocalMemorySize(device))/(8));
                    break;
                case RIGIDBODY:
                	optimalMultiples[KERNEL_drigidBodyError] = uniformBSplineTransformProgramKernels[KERNEL_drigidBodyError].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_drigidBodyErrorWithGradAndHess] = uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHess].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_dsumInLocalMemoryCombined] = uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemoryCombined].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_drigidBodyErrorWithGradAndHessBrent] = uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHessBrent].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_dtransformImageWithBsplineInterpolation] = uniformBSplineTransformProgramKernels[KERNEL_dtransformImageWithBsplineInterpolation].getPreferredWorkGroupSizeMultiple(device);
                	
                    doubleblockSizesRigidBodyParallel = (int)Math.min(Math.min(uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHessBrent].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHessBrent].getLocalMemorySize(device))/((8)*11/*11 buffers are needed*/)), blocksizeMultiplier*optimalMultiples[KERNEL_drigidBodyErrorWithGradAndHessBrent]);
                    doubleblockSizesRigidBodyParallel -= (doubleblockSizesRigidBodyParallel % optimalMultiples[KERNEL_drigidBodyErrorWithGradAndHessBrent]);
                    if(doubleblockSizesRigidBodyParallel == 0)
                    {
                        doubleblockSizesRigidBodyParallel = 1;// Fallback solution minimum is 1
                    }
                    
                    doublemaximumElementsForLocalFPTcombinedSum = (int) Math.min(uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemoryCombined].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemoryCombined].getLocalMemorySize(device))/(8));
                    break;
                case SCALEDROTATION:
                    break;
                case AFFINE:
                    break;
                }
                
                doublemaximumElementsForLocalFPTsum = (int) Math.min(uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory].getLocalMemorySize(device))/(8));
                doubleblockSizesFPT = (int)Math.min(Math.min(uniformBSplineTransformProgramKernels[KERNEL_dparallelGroupedSumReduction].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_dparallelGroupedSumReduction].getLocalMemorySize(device))/(8)), blocksizeMultiplier*optimalMultiples[KERNEL_dparallelGroupedSumReduction]);
                doubleblockSizesFPT -= (doubleblockSizesFPT % optimalMultiples[KERNEL_dparallelGroupedSumReduction]);
                if(doubleblockSizesFPT == 0)
                {
                    doubleblockSizesFPT = 1;// Fallback solution minimum is 1
                }
            }
            // these are all just the float versions
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXlp] = uniformBSplineTransformProgram.createCLKernel("fCubicBSplinePrefilter2DXlp");
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYlp] = uniformBSplineTransformProgram.createCLKernel("fCubicBSplinePrefilter2DYlp");
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhp] = uniformBSplineTransformProgram.createCLKernel("fBasicToCardinal2DXhp");
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhp] = uniformBSplineTransformProgram.createCLKernel("fBasicToCardinal2DYhp");
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp] = uniformBSplineTransformProgram.createCLKernel("fCubicBSplinePrefilter2DDeg7premulhp");
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXDeg7lp] = uniformBSplineTransformProgram.createCLKernel("fCubicBSplinePrefilter2DXDeg7lp");
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYDeg7lp] = uniformBSplineTransformProgram.createCLKernel("fCubicBSplinePrefilter2DYDeg7lp");
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhpDeg7] = uniformBSplineTransformProgram.createCLKernel("fBasicToCardinal2DXhpDeg7");
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhpDeg7] = uniformBSplineTransformProgram.createCLKernel("fBasicToCardinal2DYhpDeg7");
            uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX] = uniformBSplineTransformProgram.createCLKernel("freduceDual1DX");
            uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DY] = uniformBSplineTransformProgram.createCLKernel("freduceDual1DY");
            uniformBSplineTransformProgramKernels[KERNEL_antiSymmetricFirMirrorOffBounds1DX] = uniformBSplineTransformProgram.createCLKernel("fantiSymmetricFirMirrorOffBounds1DX");
            uniformBSplineTransformProgramKernels[KERNEL_antiSymmetricFirMirrorOffBounds1DY] = uniformBSplineTransformProgram.createCLKernel("fantiSymmetricFirMirrorOffBounds1DY");
            uniformBSplineTransformProgramKernels[KERNEL_TargetedCubicBSplinePrefilter2Dpremulhp] = uniformBSplineTransformProgram.createCLKernel("fTargetedCubicBSplinePrefilter2Dpremulhp");
            uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory] = uniformBSplineTransformProgram.createCLKernel("fsumInLocalMemory");
            uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction] = uniformBSplineTransformProgram.createCLKernel("fparallelGroupedSumReduction");
            
            switch(sharedContext.transformationType) {
            case TRANSLATION:
                uniformBSplineTransformProgramKernels[KERNEL_ftranslationError] = uniformBSplineTransformProgram.createCLKernel("ftranslationError");
                uniformBSplineTransformProgramKernels[KERNEL_ftranslationErrorWithGradAndHess] = uniformBSplineTransformProgram.createCLKernel("ftranslationErrorWithGradAndHess");
                uniformBSplineTransformProgramKernels[KERNEL_ftranslationSumInLocalMemoryCombined] = uniformBSplineTransformProgram.createCLKernel("ftranslationSumInLocalMemoryCombined");
                uniformBSplineTransformProgramKernels[KERNEL_ftranslationErrorWithGradAndHessBrent] = uniformBSplineTransformProgram.createCLKernel("ftranslationErrorWithGradAndHessBrent");
                break;
            case RIGIDBODY:
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyError] = uniformBSplineTransformProgram.createCLKernel("frigidBodyError");
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHess] = uniformBSplineTransformProgram.createCLKernel("frigidBodyErrorWithGradAndHess");
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemoryCombined] = uniformBSplineTransformProgram.createCLKernel("fsumInLocalMemoryCombined");
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent] = uniformBSplineTransformProgram.createCLKernel("frigidBodyErrorWithGradAndHessBrent");
                break;
            case SCALEDROTATION:
                break;
            case AFFINE:
                break;
            }

            /*
            The following code determines the optimal multiple size for each kernel.
            This may be different for each kernel (usually not for NVidia GPU's where
            this is either 32 or 64 but for example on Intel GPUs where unsynchronized
            kernels may be run on multiple compute devices (SIMD processors) 
            concurrently.
            */
            optimalMultiples[KERNEL_CubicBSplinePrefilter2DXlp] = uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXlp].getPreferredWorkGroupSizeMultiple(device);
            optimalMultiples[KERNEL_CubicBSplinePrefilter2DYlp] = uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYlp].getPreferredWorkGroupSizeMultiple(device);
            optimalMultiples[KERNEL_BasicToCardinal2DXhp] = uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhp].getPreferredWorkGroupSizeMultiple(device);
            optimalMultiples[KERNEL_BasicToCardinal2DYhp] = uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhp].getPreferredWorkGroupSizeMultiple(device);
            optimalMultiples[KERNEL_CubicBSplinePrefilter2DDeg7premulhp] = uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp].getPreferredWorkGroupSizeMultiple(device);
            optimalMultiples[KERNEL_CubicBSplinePrefilter2DXDeg7lp] = uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXDeg7lp].getPreferredWorkGroupSizeMultiple(device);
            optimalMultiples[KERNEL_CubicBSplinePrefilter2DYDeg7lp] = uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYDeg7lp].getPreferredWorkGroupSizeMultiple(device);
            optimalMultiples[KERNEL_BasicToCardinal2DXhpDeg7] = uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhpDeg7].getPreferredWorkGroupSizeMultiple(device);
            optimalMultiples[KERNEL_BasicToCardinal2DYhpDeg7] = uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhpDeg7].getPreferredWorkGroupSizeMultiple(device);
            optimalMultiples[KERNEL_reduceDual1DX] = uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX].getPreferredWorkGroupSizeMultiple(device);
            optimalMultiples[KERNEL_reduceDual1DY] = uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DY].getPreferredWorkGroupSizeMultiple(device);
            optimalMultiples[KERNEL_antiSymmetricFirMirrorOffBounds1DX] = uniformBSplineTransformProgramKernels[KERNEL_antiSymmetricFirMirrorOffBounds1DX].getPreferredWorkGroupSizeMultiple(device);
            optimalMultiples[KERNEL_antiSymmetricFirMirrorOffBounds1DY] = uniformBSplineTransformProgramKernels[KERNEL_antiSymmetricFirMirrorOffBounds1DY].getPreferredWorkGroupSizeMultiple(device);
            optimalMultiples[KERNEL_TargetedCubicBSplinePrefilter2Dpremulhp] = uniformBSplineTransformProgramKernels[KERNEL_TargetedCubicBSplinePrefilter2Dpremulhp].getPreferredWorkGroupSizeMultiple(device);
            optimalMultiples[KERNEL_sumInLocalMemory] = uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].getPreferredWorkGroupSizeMultiple(device);
            optimalMultiples[KERNEL_parallelGroupedSumReduction] = uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].getPreferredWorkGroupSizeMultiple(device);
            
            switch(sharedContext.transformationType) {
            case TRANSLATION:
            	optimalMultiples[KERNEL_ftranslationError] = uniformBSplineTransformProgramKernels[KERNEL_ftranslationError].getPreferredWorkGroupSizeMultiple(device);
            	optimalMultiples[KERNEL_ftranslationErrorWithGradAndHess] = uniformBSplineTransformProgramKernels[KERNEL_ftranslationErrorWithGradAndHess].getPreferredWorkGroupSizeMultiple(device);
            	optimalMultiples[KERNEL_ftranslationSumInLocalMemoryCombined] = uniformBSplineTransformProgramKernels[KERNEL_ftranslationSumInLocalMemoryCombined].getPreferredWorkGroupSizeMultiple(device);
            	optimalMultiples[KERNEL_ftranslationErrorWithGradAndHessBrent] = uniformBSplineTransformProgramKernels[KERNEL_ftranslationErrorWithGradAndHessBrent].getPreferredWorkGroupSizeMultiple(device);
                
                blockSizesRigidBodyParallel = (int)Math.min(Math.min(uniformBSplineTransformProgramKernels[KERNEL_ftranslationErrorWithGradAndHessBrent].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_ftranslationErrorWithGradAndHessBrent].getLocalMemorySize(device))/((4)*7/*we need 7 buffers*/)), blocksizeMultiplier*optimalMultiples[KERNEL_ftranslationErrorWithGradAndHessBrent]);
                blockSizesRigidBodyParallel -= (blockSizesRigidBodyParallel % optimalMultiples[KERNEL_ftranslationErrorWithGradAndHessBrent]);
                if(blockSizesRigidBodyParallel == 0)
                {
                    blockSizesRigidBodyParallel = 1;// Fallback solution minimum is 1
                }
                
                maximumElementsForLocalFPTcombinedSum = (int) Math.min(uniformBSplineTransformProgramKernels[KERNEL_ftranslationSumInLocalMemoryCombined].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_ftranslationSumInLocalMemoryCombined].getLocalMemorySize(device))/(4));
                break;
            case RIGIDBODY:
            	optimalMultiples[KERNEL_rigidBodyError] = uniformBSplineTransformProgramKernels[KERNEL_rigidBodyError].getPreferredWorkGroupSizeMultiple(device);
            	optimalMultiples[KERNEL_rigidBodyErrorWithGradAndHess] = uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHess].getPreferredWorkGroupSizeMultiple(device);
            	optimalMultiples[KERNEL_sumInLocalMemoryCombined] = uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemoryCombined].getPreferredWorkGroupSizeMultiple(device);
            	optimalMultiples[KERNEL_rigidBodyErrorWithGradAndHessBrent] = uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent].getPreferredWorkGroupSizeMultiple(device);
                
                blockSizesRigidBodyParallel = (int)Math.min(Math.min(uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent].getLocalMemorySize(device))/((4)*11/*we need 11 buffers*/)), blocksizeMultiplier*optimalMultiples[KERNEL_rigidBodyErrorWithGradAndHessBrent]);
                blockSizesRigidBodyParallel -= (blockSizesRigidBodyParallel % optimalMultiples[KERNEL_rigidBodyErrorWithGradAndHessBrent]);
                if(blockSizesRigidBodyParallel == 0)
                {
                    blockSizesRigidBodyParallel = 1;// Fallback solution minimum is 1
                }
                
                maximumElementsForLocalFPTcombinedSum = (int) Math.min(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemoryCombined].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemoryCombined].getLocalMemorySize(device))/(4));
                break;
            case SCALEDROTATION:
                break;
            case AFFINE:
                break;
            }

            maximumElementsForLocalFPTsum = (int) Math.min(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].getLocalMemorySize(device))/(4));
            blockSizesFPT = (int)Math.min(Math.min(uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].getLocalMemorySize(device))/(4)), blocksizeMultiplier*optimalMultiples[KERNEL_parallelGroupedSumReduction]);
            blockSizesFPT -= (blockSizesFPT % optimalMultiples[KERNEL_parallelGroupedSumReduction]);
            if(blockSizesFPT == 0)
            {
                blockSizesFPT = 1;// Fallback solution minimum is 1
            }
            
            if(!usesFloat)
            {
                // only useful if the GPU is capable of accepting double
                // Ugly code but I didn't figure out how to do it more elegantly
                if(sharedContext.img.firstElement() instanceof ByteType)
                {
                    conversionProgram = context.createProgram(System.class.getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/Conversion.cl")).build("-D SDT=char -D CMD=convert_char_sat_rte -D TDT=double -D MAXVAL=\"127.0\"",devices);
                    conversionProgramKernel = conversionProgram.createCLKernel("Convert");
                    conversionProgramKernel.putArg(conversionEntryBuffer).putArg(doubleEntryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                    deConversionProgramKernel = conversionProgram.createCLKernel("deConvert");
                    deConversionProgramKernel.putArg(doubleFullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                }
                if(sharedContext.img.firstElement() instanceof UnsignedByteType)
                {
                    conversionProgram = context.createProgram(System.class.getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/Conversion.cl")).build("-D SDT=uchar -D CMD=convert_uchar_sat_rte -D TDT=double -D MAXVAL=\"255.0\"",devices);
                    conversionProgramKernel = conversionProgram.createCLKernel("Convert");
                    conversionProgramKernel.putArg(conversionEntryBuffer).putArg(doubleEntryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                    deConversionProgramKernel = conversionProgram.createCLKernel("deConvert");
                    deConversionProgramKernel.putArg(doubleFullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                }
                else if((sharedContext.img.firstElement() instanceof ShortType))
                {
                    conversionProgram = context.createProgram(System.class.getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/Conversion.cl")).build("-D SDT=short -D CMD=convert_short_sat_rte -D TDT=double -D MAXVAL=\"32767.0\"",devices);
                    conversionProgramKernel = conversionProgram.createCLKernel("Convert");
                    conversionProgramKernel.putArg(conversionEntryBuffer).putArg(doubleEntryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                    deConversionProgramKernel = conversionProgram.createCLKernel("deConvert");
                    deConversionProgramKernel.putArg(doubleFullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                }
                else if((sharedContext.img.firstElement() instanceof UnsignedShortType))
                {
                    conversionProgram = context.createProgram(System.class.getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/Conversion.cl")).build("-D SDT=ushort -D CMD=convert_ushort_sat_rte -D TDT=double -D MAXVAL=\"65535.0\"",devices);
                    conversionProgramKernel = conversionProgram.createCLKernel("Convert");
                    conversionProgramKernel.putArg(conversionEntryBuffer).putArg(doubleEntryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                    deConversionProgramKernel = conversionProgram.createCLKernel("deConvert");
                    deConversionProgramKernel.putArg(doubleFullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                }
                else if((sharedContext.img.firstElement() instanceof IntType))
                {
                    conversionProgram = context.createProgram(System.class.getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/Conversion.cl")).build("-D SDT=int -D CMD=convert_int_sat_rte -D TDT=double -D MAXVAL=\"2147483647.0\"",devices);
                    conversionProgramKernel = conversionProgram.createCLKernel("Convert");
                    conversionProgramKernel.putArg(conversionEntryBuffer).putArg(doubleEntryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                    deConversionProgramKernel = conversionProgram.createCLKernel("deConvert");
                    deConversionProgramKernel.putArg(doubleFullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                }
                else if((sharedContext.img.firstElement() instanceof UnsignedIntType))
                {
                    conversionProgram = context.createProgram(System.class.getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/Conversion.cl")).build("-D SDT=uint -D CMD=convert_uint_sat_rte -D TDT=double -D MAXVAL=\"4294967295.0\"",devices);
                    conversionProgramKernel = conversionProgram.createCLKernel("Convert");
                    conversionProgramKernel.putArg(conversionEntryBuffer).putArg(doubleEntryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                    deConversionProgramKernel = conversionProgram.createCLKernel("deConvert");
                    deConversionProgramKernel.putArg(doubleFullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                }
                else if((sharedContext.img.firstElement() instanceof FloatType))
                {
                    conversionProgram = context.createProgram(System.class.getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/Conversion.cl")).build("-D SDT=float -D CMD=convert_float_sat_rte -D TDT=double",devices);
                    conversionProgramKernel = conversionProgram.createCLKernel("Convert");
                    conversionProgramKernel.putArg(conversionEntryBuffer).putArg(doubleEntryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                    deConversionProgramKernel = conversionProgram.createCLKernel("deConvert");
                    deConversionProgramKernel.putArg(doubleFullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                }
            }
        }
        
        @Override
        public void run() {
            while(!sharedContext.getNextAlignmentTarget(scat))
            {                
                putTargetImageIntoPipelineEntry();
                // Note that the image is (also converted if necessary) in entryImageBuffers[id], the pyramid can now be constructed
                constructTargetImagePyramid();
                // Now put the source image into the pipeline (entryImageBuffer) and convert it if necessary
                putSourceImageIntoPipelineEntry();
                // Construct the image and derivative pyramids
                constructSourceImagePyramid();
                // Now that the pyramid is constructed the optimization can commence
                doRegistration();
                switch(sharedContext.transformationType)
                {
                    case TRANSLATION:
                        ((TranslationTransformation)scat.transformation).offsetx = offsetx;
                        ((TranslationTransformation)scat.transformation).offsety = offsety;
                        break;
                    case RIGIDBODY:
                        ((RigidBodyTransformation)scat.transformation).angle = angle;
                        ((RigidBodyTransformation)scat.transformation).offsetx = offsetx;
                        ((RigidBodyTransformation)scat.transformation).offsety = offsety;
                        break;
                    case SCALEDROTATION:
                        break;
                    case AFFINE:
                        break;
                }
                
                // reset the transformation values for the next image
                offsetx = 0.0;
                offsety = 0.0;
                angle = 0.0;
                queue.finish();
            }
            try {
                // Done calculating the transformations now wait until all threads are done
                sharedContext.workerSynchronizationBarrier.await(); // This also combines all the transformations and resets the position (see the action implementation)
            } catch (InterruptedException ex) {
                throw new RuntimeException("Thread interrupted.");
            } catch (BrokenBarrierException ex) {
                throw new RuntimeException("Worker synchronization barrier is broken.");
            }
            
            if(sharedContext instanceof SharedContextZT) 
            {
                while(!sharedContext.getNextAlignmentTarget(scat))
                {                
                    putTargetImageIntoPipelineEntry();
                    // Note that the image is (also converted if necessary) in entryImageBuffers[id], the pyramid can now be constructed
                    constructTargetImagePyramid();
                    // Now put the source image into the pipeline (entryImageBuffer) and convert it if necessary
                    putSourceImageIntoPipelineEntry();
                    // Construct the image and derivative pyramids
                    constructSourceImagePyramid();
                    // Now that the pyramid is constructed the optimization can commence
                    doRegistration();
                    switch(sharedContext.transformationType)
                    {
                        case TRANSLATION:
                            ((TranslationTransformation)scat.transformation).offsetx = offsetx;
                            ((TranslationTransformation)scat.transformation).offsety = offsety;
                            break;
                        case RIGIDBODY:
                            ((RigidBodyTransformation)scat.transformation).angle = angle;
                            ((RigidBodyTransformation)scat.transformation).offsetx = offsetx;
                            ((RigidBodyTransformation)scat.transformation).offsety = offsety;
                            break;
                        case SCALEDROTATION:
                            break;
                        case AFFINE:
                            break;
                    }
                    
                    // reset the transformation values for the next image
                    offsetx = 0.0;
                    offsety = 0.0;
                    angle = 0.0;
                    queue.finish();
                }
                try {
                    // Done calculating the transformations now wait until all threads are done
                    sharedContext.workerSynchronizationBarrier.await(); // This also combines all the transformations and resets the position (see the action implementation)
                } catch (InterruptedException ex) {
                    throw new RuntimeException("Thread interrupted.");
                } catch (BrokenBarrierException ex) {
                    throw new RuntimeException("Worker synchronization barrier is broken.");
                }
            }
            
            // The following code requires the current position to have been reset to 0vector 
            while(!sharedContext.getTransformationForCurrentPosition(scat))
            {
                putTargetImageIntoPipelineEntry();// This copies and converts the image to the GPU
                switch(sharedContext.transformationType)
                {
                    case TRANSLATION:
                        transformImageTranslation();
                        break;
                    case RIGIDBODY:
                        transformImageRigidBody();
                        break;
                    case SCALEDROTATION:
                        break;
                    case AFFINE:
                        break;
                }
                queue.finish();
            }
        }
        
        private void convertToBSplineCoeff(final int width, final int height, int localWorkSize, int globalWorkSize)
        {
        	// pre-multiply the image
            uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2Dpremulhp].putArg(doubleEntryImageBuffer).putArg(width*height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2Dpremulhp],0,globalWorkSize,localWorkSize);
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DXhp].getWorkGroupSize(device);
            uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DXhp].putArg(doubleEntryImageBuffer).putArg(width).putArg(height);
            // Conversion to B-spline coefficients along X axis (Group size must be >= height)
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dCubicBSplinePrefilter2DXhp], height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DXhp],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DXhp].rewind();
            // Now along Y-axis
            // Has to be pre-multiplied by lambda again!!!
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2Dpremulhp].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dCubicBSplinePrefilter2Dpremulhp], width*height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2Dpremulhp],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2Dpremulhp].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DYhp].putArg(doubleEntryImageBuffer).putArg(width).putArg(height);
            // Conversion to B-spline coefficients along Y axis (Group size must be >= width)
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DYhp].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dCubicBSplinePrefilter2DYhp], width);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DYhp],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DYhp].rewind();
        }
        
        private void convertToBSplineCoeffCPU(final int width, final int height)
        {
            converter.convertTo(scat.targetArray, targetCoefficientDoubleSlice);
            // pre-multiply the image for cubic spline interpolation
            PlainJavaCPUAligner.premultiplyCubicBSpline(targetCoefficientDoubleSlice, width * height);
            // Conversion to B-spline coefficients along X axis
            PlainJavaCPUAligner.cubicBSplinePrefilter2DXhp(targetCoefficientDoubleSlice, width, height);
            // pre-multiply again
            PlainJavaCPUAligner.premultiplyCubicBSpline(targetCoefficientDoubleSlice, width * height);
            // Now along Y-axis
            PlainJavaCPUAligner.cubicBSplinePrefilter2DYhp(targetCoefficientDoubleSlice, width, height);
        }
        
        private void transformImageRigidBody()
        {
            int width = (int)sharedContext.img.dimension(0);
            int height = (int)sharedContext.img.dimension(1);
            // This has to be done from the double buffers
            if(!usesFloat)
            {
                // GPU
                int localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2Dpremulhp].getWorkGroupSize(device);  // Local work size dimensions
                int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dCubicBSplinePrefilter2Dpremulhp], width*height);   // rounded up to the nearest multiple of the localWorkSize
                convertToBSplineCoeff(width, height, localWorkSize, globalWorkSize);

                /*
                Now that the B-spline coefficients are in the doubleEntryImageBuffer,
                the image needs to be transformed, rescaled back and then 
                converted back to the original image format.
                */
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dtransformImageWithBsplineInterpolation].getWorkGroupSize(device);  // Local work size dimensions
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dtransformImageWithBsplineInterpolation], width*height);   // rounded up to the nearest multiple of the localWorkSize
                uniformBSplineTransformProgramKernels[KERNEL_dtransformImageWithBsplineInterpolation]
                        .putArg(doubleEntryImageBuffer)
                        .putArg(doubleFullSizedGPUResidentHelperBuffer)
                        .putArg(width)
                        .putArg(height)
                        .putArg(((RigidBodyTransformation)scat.transformation).offsetx)
                        .putArg(((RigidBodyTransformation)scat.transformation).offsety)
                        .putArg(((RigidBodyTransformation)scat.transformation).angle);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dtransformImageWithBsplineInterpolation],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_dtransformImageWithBsplineInterpolation].rewind();

                // Transformed and converted back to an image at the same time
                // now convert it back to the original format
                fetchTransformedImage();
            }
            else
            {
                // CPU
            	convertToBSplineCoeffCPU(width, height);

                transformImageWithBsplineInterpolation(width,height, ((RigidBodyTransformation)scat.transformation).offsetx, ((RigidBodyTransformation)scat.transformation).offsety, ((RigidBodyTransformation)scat.transformation).angle);

                converter.deConvertTo(CPUentryImageBuffer,scat.targetArray);
            }
        }
        
        private void transformImageTranslation()
        {
            int width = (int)sharedContext.img.dimension(0);
            int height = (int)sharedContext.img.dimension(1);
            // This has to be done from the double buffers
            if(!usesFloat)
            {
                // GPU
                int localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2Dpremulhp].getWorkGroupSize(device);  // Local work size dimensions
                int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dCubicBSplinePrefilter2Dpremulhp], width*height);   // rounded up to the nearest multiple of the localWorkSize
                convertToBSplineCoeff(width, height, localWorkSize, globalWorkSize);

                /*
                Now that the B-spline coefficients are in the doubleEntryImageBuffer,
                the image needs to be transformed, rescaled back and then 
                converted back to the original image format.
                */
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dtranslationtransformImageWithBsplineInterpolation].getWorkGroupSize(device);  // Local work size dimensions
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dtranslationtransformImageWithBsplineInterpolation], width*height);   // rounded up to the nearest multiple of the localWorkSize
                uniformBSplineTransformProgramKernels[KERNEL_dtranslationtransformImageWithBsplineInterpolation]
                        .putArg(doubleEntryImageBuffer)
                        .putArg(doubleFullSizedGPUResidentHelperBuffer)
                        .putArg(width)
                        .putArg(height)
                        .putArg(((TranslationTransformation)scat.transformation).offsetx)
                        .putArg(((TranslationTransformation)scat.transformation).offsety);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dtransformImageWithBsplineInterpolation],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_dtransformImageWithBsplineInterpolation].rewind();

                // Transformed and converted back to an image at the same time
                // now convert it back to the original format
                fetchTransformedImage();
            }
            else
            {
                // CPU
            	convertToBSplineCoeffCPU(width, height);

                translationTransformImageWithBsplineInterpolation(width,height, ((TranslationTransformation)scat.transformation).offsetx, ((TranslationTransformation)scat.transformation).offsety);

                converter.deConvertTo(CPUentryImageBuffer,scat.targetArray);
            }
        }
        
        private void transformImageWithBsplineInterpolation(final int width, final int height, double currentoffsetx, double currentoffsety, double currentangle)
        {
            /*
            Requires the coefficients to be in entryImageBuffers and the output
            will be in fullSizedHelperBuffer
            */
            int doubleWidth = width*2;
            int doubleHeight = height*2;
            int nIndex = 0;
            int p;
            int q;
            int tmpindex;
            double s;
            double xvecx = Math.cos(currentangle);
            double xvecy = -Math.sin(currentangle);
            double yvecx = -xvecy;
            double yvecy = xvecx;
            double coordx;
            double rescoordx;
            double coordy;
            double rescoordy;
            int mskx;
            int msky;
            for(int i = 0;i < height;i++)
            {
            	// First walk along the Y-vector direction and reset the X-position (otherwise the
                // y position is initially correct and then lagging behind by one all the time)
                coordx = currentoffsetx + ((double)i) * yvecx;
                coordy = currentoffsety + ((double)i) * yvecy;
                for(int n = 0;n < width;n++,nIndex++)
                {
                    mskx = (int)Math.round(coordx);
                    msky = (int)Math.round(coordy);
                    if((mskx >= 0)&&(mskx < width)&&(msky >= 0)&&(msky < height))
                    {
                        // Calculate X-interpolation indices
                        p = (coordx >= 0) ? (((int)coordx) + 2) : (((int)coordx) + 1);
                        for(int c = 0;c < 4;c++,p--)
                        {
                            q = (p < 0) ? (-1 - p) : p;
                            if(q >= doubleWidth)
                            {
                                q -= (doubleWidth) * (q / (doubleWidth)); // this is an integer division it doesn't yield q
                            }
                            xInterpolationIndices[c] = q >= width ? (doubleWidth - 1 - q) : q;
                        }
                        // calculate Y-interpolation indices
                        p = (coordy >= 0) ? (((int)coordy) + 2) : (((int)coordy) + 1);
                        for(int c = 0;c < 4;c++,p--)
                        {
                            q = (p < 0) ? (-1 - p) : p;
                            if(q >= doubleHeight)
                            {
                                q -= (doubleHeight) * (q / (doubleHeight)); // this is an integer division it doesn't yield q
                            }
                            yInterpolationIndices[c] = q >= height ? (doubleHeight - 1 - q) * width : q * width;// calculate linearized coordinates of the coefficient array
                        }
                        // get the residuals of the coordinates
                        rescoordx = coordx - (coordx >= 0.0 ? ((double)((int)coordx)) : ((double)(((int)coordx) - 1)));
                        rescoordy = coordy - (coordy >= 0.0 ? ((double)((int)coordy)) : ((double)(((int)coordy) - 1)));
                        // calculate the X-weights
                        s = 1.0 - rescoordx;
                        xWeights[3] = Math.pow(s, 3.0) / 6.0;
                        s = rescoordx * rescoordx;
                        xWeights[2] = (2.0 / 3.0) - 0.5 * s * (2.0 - rescoordx);
                        xWeights[0] = s * rescoordx / 6.0;
                        xWeights[1] = 1.0 - xWeights[0] - xWeights[2] - xWeights[3];
                        // calculate the Y-weights
                        s = 1.0 - rescoordy;
                        yWeights[3] = Math.pow(s, 3.0) / 6.0;
                        s = rescoordy * rescoordy;
                        yWeights[2] = (2.0 / 3.0) - 0.5 * s * (2.0 - rescoordy);
                        yWeights[0] = s * rescoordy / 6.0;
                        yWeights[1] = 1.0 - yWeights[0] - yWeights[2] - yWeights[3];

                        // calculate the interpolated value from the target coefficients
                        s = 0.0;
                        for(int y = 0;y < 4;y++)
                        {
                            rescoordx = 0.0;// To avoid using too many variables this one will be repurposed
                            tmpindex = yInterpolationIndices[y];
                            for(int x = 0;x < 4; x++)
                            {
                                rescoordx += xWeights[x] * targetCoefficientDoubleSlice[tmpindex + xInterpolationIndices[x]];
                            }
                            s += yWeights[y] * rescoordx;
                        }
                        CPUentryImageBuffer[nIndex] = s;
                    }
                    else
                    {
                        CPUentryImageBuffer[nIndex] = 0.0;
                    }
                    // walk along the X-vector direction
                    coordx += xvecx;
                    coordy += xvecy;
                }
            }
        }
        
        private void translationTransformImageWithBsplineInterpolation(final int width, final int height, double currentoffsetx, double currentoffsety)
        {
            /*
            Requires the coefficients to be in entryImageBuffers and the output
            will be in fullSizedHelperBuffer
            */
            int doubleWidth = width*2;
            int doubleHeight = height*2;
            int nIndex = 0;
            int p;
            int q;
            int tmpindex;
            double s;
            double coordx;
            double rescoordx;
            double coordy;
            double rescoordy;
            int mskx;
            int msky;
            for(int i = 0;i < height;i++)
            {
            	// First walk along the Y-vector direction and reset the X-position (otherwise the
                // y position is initially correct and then lagging behind by one all the time)
                coordx = currentoffsetx;
                coordy = currentoffsety + ((double)i);
                for(int n = 0;n < width;n++,nIndex++)
                {
                    mskx = (int)Math.round(coordx);
                    msky = (int)Math.round(coordy);
                    if((mskx >= 0)&&(mskx < width)&&(msky >= 0)&&(msky < height))
                    {
                        // Calculate X-interpolation indices
                        p = (coordx >= 0) ? (((int)coordx) + 2) : (((int)coordx) + 1);
                        for(int c = 0;c < 4;c++,p--)
                        {
                            q = (p < 0) ? (-1 - p) : p;
                            if(q >= doubleWidth)
                            {
                                q -= (doubleWidth) * (q / (doubleWidth)); // Warning this is an integer division it doesn't yield q
                            }
                            xInterpolationIndices[c] = q >= width ? (doubleWidth - 1 - q) : q;
                        }
                        // calculate Y-interpolation indices
                        p = (coordy >= 0) ? (((int)coordy) + 2) : (((int)coordy) + 1);
                        for(int c = 0;c < 4;c++,p--)
                        {
                            q = (p < 0) ? (-1 - p) : p;
                            if(q >= doubleHeight)
                            {
                                q -= (doubleHeight) * (q / (doubleHeight)); // Warning this is an integer division it doesn't yield q
                            }
                            yInterpolationIndices[c] = q >= height ? (doubleHeight - 1 - q) * width : q * width;// calculate linearized coordinates of the coefficient array
                        }
                        // get the residuals of the coordinates
                        rescoordx = coordx - (coordx >= 0.0 ? ((double)((int)coordx)) : ((double)(((int)coordx) - 1)));
                        rescoordy = coordy - (coordy >= 0.0 ? ((double)((int)coordy)) : ((double)(((int)coordy) - 1)));
                        // calculate the X-weights
                        s = 1.0 - rescoordx;
                        xWeights[3] = Math.pow(s, 3.0) / 6.0;
                        s = rescoordx * rescoordx;
                        xWeights[2] = (2.0 / 3.0) - 0.5 * s * (2.0 - rescoordx);
                        xWeights[0] = s * rescoordx / 6.0;
                        xWeights[1] = 1.0 - xWeights[0] - xWeights[2] - xWeights[3];
                        // calculate the Y-weights
                        s = 1.0 - rescoordy;
                        yWeights[3] = Math.pow(s, 3.0) / 6.0;
                        s = rescoordy * rescoordy;
                        yWeights[2] = (2.0 / 3.0) - 0.5 * s * (2.0 - rescoordy);
                        yWeights[0] = s * rescoordy / 6.0;
                        yWeights[1] = 1.0 - yWeights[0] - yWeights[2] - yWeights[3];

                        // calculate the interpolated value from the target coefficients
                        s = 0.0;
                        for(int y = 0;y < 4;y++)
                        {
                            rescoordx = 0.0;// To avoid using too many variables this one will be repurposed
                            tmpindex = yInterpolationIndices[y];
                            for(int x = 0;x < 4; x++)
                            {
                                rescoordx += xWeights[x] * targetCoefficientDoubleSlice[tmpindex + xInterpolationIndices[x]];
                            }
                            s += yWeights[y] * rescoordx;
                        }
                        CPUentryImageBuffer[nIndex] = s;
                    }
                    else
                    {
                        CPUentryImageBuffer[nIndex] = 0.0;
                    }
                    // walk along the X-vector direction
                    coordx += 1.0;
                }
            }
        }
        
        private void fetchTransformedImage()
        {
            // This function is only called from the GPU code so one can assume !usesFloat == true
            // WARNING: this function overwrites the original image data without a chance to recover it!!!
           
            // The following converts the image to its original format and transfers it to conversionEntryBuffer
            int localWorkSize = (int)deConversionProgramKernel.getWorkGroupSize(device);  // Local work size dimensions
            int globalWorkSize = StaticUtility.roundUp(localWorkSize, deConversionProgramKernel.getPreferredWorkGroupSizeMultiple(device),(int) sharedContext.img.dimension(0)*(int)sharedContext.img.dimension(1));   // rounded up to the nearest multiple of the localWorkSize
            queue.put1DRangeKernel(deConversionProgramKernel,0,globalWorkSize,localWorkSize);
            if ((sharedContext.img.firstElement() instanceof ByteType)||(sharedContext.img.firstElement() instanceof UnsignedByteType)) {
                ByteBuffer buffer = queue.putMapBuffer(conversionEntryBuffer, CLMemory.Map.READ, true);
                buffer.rewind();
                buffer.get((byte[]) scat.targetArray);
                queue.putUnmapMemory(conversionEntryBuffer, buffer);
            } else if ((sharedContext.img.firstElement() instanceof ShortType)||(sharedContext.img.firstElement() instanceof UnsignedShortType)) {
                ByteBuffer buffer = queue.putMapBuffer(conversionEntryBuffer, CLMemory.Map.READ, true);
                buffer.rewind();
                buffer.asShortBuffer().get((short[]) scat.targetArray);
                queue.putUnmapMemory(conversionEntryBuffer, buffer);
            } else if ((sharedContext.img.firstElement() instanceof IntType)||(sharedContext.img.firstElement() instanceof UnsignedIntType)) {
                ByteBuffer buffer = queue.putMapBuffer(conversionEntryBuffer, CLMemory.Map.READ, true);
                buffer.rewind();
                buffer.asIntBuffer().get((int[]) scat.targetArray);
                queue.putUnmapMemory(conversionEntryBuffer, buffer);
            } else if ((sharedContext.img.firstElement() instanceof FloatType)) {
                ByteBuffer buffer = queue.putMapBuffer(conversionEntryBuffer, CLMemory.Map.READ, true);
                buffer.rewind();
                buffer.asFloatBuffer().get((float[]) scat.targetArray);
                queue.putUnmapMemory(conversionEntryBuffer, buffer);
            }
        }
        
        
        private void doRegistration()
        {
            iterationPower = (int)Math.pow(2.0, (double)pyramidDepth);
            
            switch(sharedContext.transformationType) {
            case TRANSLATION:
                for(int i = pyramidDepth - 1;i > 0;i--)
                {
                    iterationPower /= 2;
                    inverseMarquardtLevenbergTranslationOptimization(i);
                    // scale up
                    offsetx *= 2.0;
                    offsety *= 2.0;
                }
                iterationPower /= 2;
                inverseMarquardtLevenbergTranslationOptimization(0);
                
                // With the perfect initial guess of the parameters run one last round at double precision
                doubleInverseMarquardtLevenbergTranslationOptimization();
                break;
            case RIGIDBODY:
                for(int i = pyramidDepth - 1;i > 0;i--)
                {
                    iterationPower /= 2;
                    inverseMarquardtLevenbergRigidBodyOptimization(i);
                    // scale up (but the rotation is not scale dependent so simply scale up the translation)
                    offsetx *= 2.0;
                    offsety *= 2.0;
                }
                iterationPower /= 2;
                inverseMarquardtLevenbergRigidBodyOptimization(0);
                
                // With the perfect initial guess of the parameters run one last round at double precision
                doubleInverseMarquardtLevenbergRigidBodyOptimization();
                break;
            case SCALEDROTATION:
                break;
            case AFFINE:
                break;
            }
        }

        private void inverseMarquardtLevenbergRigidBodyOptimization(int pyramidIndex)
        {
            double[] update = {0.0,0.0,0.0};
            double bestMeanSquares = 0.0;
            double meanSquares = 0.0;
            double lambda = 1.0;
            double displacement;
            int iteration = 0;
            double c;
            double s;
            // first initialize the matrix with the current transformation (upscaling between the steps)
            double currentoffsetx;
            double currentoffsety;
            double currentangle;
            bestMeanSquares = getRigidBodyMeanSquares(pyramidIndex,offsetx,offsety,this.angle);
            iteration++;
            do {
                // calculate the pseudo hessian from the hessian
                for (int k = 0; (k < 3); k++) {
                        pseudoHessian[k][k] = (1.0 + lambda) * hessian[k][k];
                }
                StaticUtility.invertGauss(pseudoHessian);
                update = StaticUtility.matrixMultiply(pseudoHessian, gradient);
                currentangle = this.angle - update[0];
                displacement = Math.sqrt(update[1] * update[1] + update[2] * update[2]) + 0.25 * Math.sqrt((double)(targetPyramid[pyramidIndex].width * targetPyramid[pyramidIndex].width) + (double)(targetPyramid[pyramidIndex].height * targetPyramid[pyramidIndex].height)) * Math.abs(update[0]);
                c = Math.cos(update[0]);
                s = Math.sin(update[0]);
                currentoffsetx = (offsetx + update[1]) * c - (offsety + update[2]) * s;
                currentoffsety = (offsetx + update[1]) * s + (offsety + update[2]) * c;
                meanSquares = getRigidBodyMeanSquares(pyramidIndex,currentoffsetx,currentoffsety,currentangle);

                iteration++;
                if (meanSquares < bestMeanSquares) {
                        bestMeanSquares = meanSquares;
                        lambda /= 4.0;
                        offsetx = currentoffsetx;
                        offsety = currentoffsety;
                        this.angle = currentangle;
                }
                else {
                        lambda *= 4.0;
                }
            } while ((iteration < (10 * iterationPower - 1)) && (0.001 <= displacement));
            StaticUtility.invertGauss(hessian);
            update = StaticUtility.matrixMultiply(hessian, gradient);
            currentangle = this.angle - update[0];
            c = Math.cos(update[0]);
            s = Math.sin(update[0]);
            currentoffsetx = (offsetx + update[1]) * c  - (offsety + update[2]) * s;
            currentoffsety = (offsetx + update[1]) * s  + (offsety + update[2]) * c;
            meanSquares = getRigidBodyMeanSquaresWithoutHessian(pyramidIndex,currentoffsetx,currentoffsety,currentangle);
            iteration++;
            if (meanSquares < bestMeanSquares) {
                offsetx = currentoffsetx;
                offsety = currentoffsety;
                this.angle = currentangle;
            }
        }
        
        
        private void inverseMarquardtLevenbergTranslationOptimization(int pyramidIndex)
        {
            double[] update = {0.0,0.0};
            double bestMeanSquares = 0.0;
            double meanSquares = 0.0;
            double lambda = 1.0;
            double displacement;
            int iteration = 0;
            double c;
            double s;
            // first initialize the matrix with the current transformation (upscaling between the steps)
            double currentoffsetx;
            double currentoffsety;
            bestMeanSquares = getTranslationMeanSquares(pyramidIndex,offsetx,offsety);
            iteration++;
            do {
                // calculate the pseudo hessian from the hessian
                for (int k = 0; (k < 2); k++) {
                        pseudoHessian[k][k] = (1.0 + lambda) * hessian[k][k];
                }
                StaticUtility.invertGauss(pseudoHessian);
                update = StaticUtility.matrixMultiply(pseudoHessian, gradient);
                displacement = Math.sqrt(update[0] * update[0] + update[1] * update[1]);
                currentoffsetx = offsetx + update[0];
                currentoffsety = offsety + update[1];
                meanSquares = getTranslationMeanSquares(pyramidIndex,currentoffsetx,currentoffsety);

                iteration++;
                if (meanSquares < bestMeanSquares) {
                        bestMeanSquares = meanSquares;
                        lambda /= 4.0;
                        offsetx = currentoffsetx;
                        offsety = currentoffsety;
                }
                else {
                        lambda *= 4.0;
                }
            } while ((iteration < (10 * iterationPower - 1)) && (0.001 <= displacement));
            StaticUtility.invertGauss(hessian);
            update = StaticUtility.matrixMultiply(hessian, gradient);
            currentoffsetx = offsetx + update[0];
            currentoffsety = offsety + update[1];
            meanSquares = getTranslationMeanSquaresWithoutHessian(pyramidIndex,currentoffsetx,currentoffsety);
            iteration++;
            if (meanSquares < bestMeanSquares) {
                offsetx = currentoffsetx;
                offsety = currentoffsety;
            }
        }
        
        
        private void doubleInverseMarquardtLevenbergRigidBodyOptimization()
        {
            double[] update = {0.0,0.0,0.0};
            double bestMeanSquares = 0.0;
            double meanSquares = 0.0;
            double lambda = 1.0;
            double displacement;
            int iteration = 0;
            double c;
            double s;
            // first initialize the matrix with the current transformation (upscaling between the steps)
            double currentoffsetx;
            double currentoffsety;
            double currentangle;
            bestMeanSquares = doubleGetRigidBodyMeanSquares(offsetx,offsety,this.angle);
            iteration++;
            do {
                // calculate the pseudo hessian from the hessian
                for (int k = 0; (k < 3); k++) {
                        pseudoHessian[k][k] = (1.0 + lambda) * hessian[k][k];
                }
                StaticUtility.invertGauss(pseudoHessian);
                update = StaticUtility.matrixMultiply(pseudoHessian, gradient);
                currentangle = this.angle - update[0];
                displacement = Math.sqrt(update[1] * update[1] + update[2] * update[2]) + 0.25 * Math.sqrt((double)(targetPyramid[0].width * targetPyramid[0].width) + (double)(targetPyramid[0].height * targetPyramid[0].height)) * Math.abs(update[0]);
                c = Math.cos(update[0]);
                s = Math.sin(update[0]);
                currentoffsetx = (offsetx + update[1]) * c - (offsety + update[2]) * s;
                currentoffsety = (offsetx + update[1]) * s + (offsety + update[2]) * c;
                meanSquares = doubleGetRigidBodyMeanSquares(currentoffsetx,currentoffsety,currentangle);

                iteration++;
                if (meanSquares < bestMeanSquares) {
                        bestMeanSquares = meanSquares;
                        lambda /= 4.0;
                        offsetx = currentoffsetx;
                        offsety = currentoffsety;
                        this.angle = currentangle;
                }
                else {
                        lambda *= 4.0;
                }
            } while ((iteration < (10 * iterationPower - 1)) && (0.001 <= displacement));
            StaticUtility.invertGauss(hessian);
            update = StaticUtility.matrixMultiply(hessian, gradient);
            currentangle = this.angle - update[0];
            c = Math.cos(update[0]);
            s = Math.sin(update[0]);
            currentoffsetx = (offsetx + update[1]) * c  - (offsety + update[2]) * s;
            currentoffsety = (offsetx + update[1]) * s  + (offsety + update[2]) * c;
            meanSquares = doubleGetRigidBodyMeanSquaresWithoutHessian(currentoffsetx,currentoffsety,currentangle);
            iteration++;
            if (meanSquares < bestMeanSquares) {
                offsetx = currentoffsetx;
                offsety = currentoffsety;
                this.angle = currentangle;
            }
        }
        
        
        private void doubleInverseMarquardtLevenbergTranslationOptimization()
        {
            double[] update = {0.0,0.0};
            double bestMeanSquares = 0.0;
            double meanSquares = 0.0;
            double lambda = 1.0;
            double displacement;
            int iteration = 0;
            double c;
            double s;
            // first initialize the matrix with the current transformation (upscaling between the steps)
            double currentoffsetx;
            double currentoffsety;
            bestMeanSquares = doubleGetTranslationMeanSquares(offsetx,offsety);
            iteration++;
            do {
                // calculate the pseudo hessian from the hessian
                for (int k = 0; (k < 2); k++) {
                        pseudoHessian[k][k] = (1.0 + lambda) * hessian[k][k];
                }
                StaticUtility.invertGauss(pseudoHessian);
                update = StaticUtility.matrixMultiply(pseudoHessian, gradient);
                displacement = Math.sqrt(update[0] * update[0] + update[1] * update[1]);
                currentoffsetx = offsetx + update[0];
                currentoffsety = offsety + update[1];
                meanSquares = doubleGetTranslationMeanSquares(currentoffsetx,currentoffsety);

                iteration++;
                if (meanSquares < bestMeanSquares) {
                        bestMeanSquares = meanSquares;
                        lambda /= 4.0;
                        offsetx = currentoffsetx;
                        offsety = currentoffsety;
                }
                else {
                        lambda *= 4.0;
                }
            } while ((iteration < (10 * iterationPower - 1)) && (0.001 <= displacement));
            StaticUtility.invertGauss(hessian);
            update = StaticUtility.matrixMultiply(hessian, gradient);
            currentoffsetx = offsetx + update[0];
            currentoffsety = offsety + update[1];
            meanSquares = doubleGetTranslationMeanSquaresWithoutHessian(currentoffsetx,currentoffsety);
            iteration++;
            if (meanSquares < bestMeanSquares) {
                offsetx = currentoffsetx;
                offsety = currentoffsety;
            }
        }
        
        private double getRigidBodyMeanSquares(int pyramidIndex, double currentoffsetx, double curentoffsety, double currentangle)
        {
            // now we have the diffs let's calculate all the derived elements
            int halfReductionSize = (int) (((sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)+((sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)%2))/2);
            ByteBuffer buffer;
            double area;
            // calculate the MSE row-wise
            double mse;
            if(halfReductionSize <= maximumElementsForLocalFPTcombinedSum)
            {                
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHess]
                        .putArg(sourcePyramid[pyramidIndex].Image)
                        .putArg(targetPyramid[pyramidIndex].Coefficient)
                        .putArg(sourcePyramid[pyramidIndex].xGradient)
                        .putArg(sourcePyramid[pyramidIndex].yGradient)
                        .putArg(gradient0)
                        .putArg(gradient1)
                        .putArg(gradient2)
                        .putArg(hessian00)
                        .putArg(hessian01)
                        .putArg(hessian02)
                        .putArg(hessian11)
                        .putArg(hessian12)
                        .putArg(hessian22)
                        .putArg(entryImageBuffer)
                        .putArg(maskBuffer)
                        .putArg((int)sourcePyramid[pyramidIndex].width)
                        .putArg((int)sourcePyramid[pyramidIndex].height)
                        .putArg((int)targetPyramid[pyramidIndex].width)
                        .putArg((int)targetPyramid[pyramidIndex].height)
                        .putArg((float)currentoffsetx)
                        .putArg((float)curentoffsety)
                        .putArg((float)currentangle);

                int localWorkSize = (int)Math.min(uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHess].getWorkGroupSize(device),blocksizeMultiplier*optimalMultiples[KERNEL_rigidBodyErrorWithGradAndHess]);  // Local work size dimensions
                int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_rigidBodyErrorWithGradAndHess], (int)(sourcePyramid[pyramidIndex].width*sourcePyramid[pyramidIndex].height));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHess],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHess].rewind();
                
                /*
                Because each of the following kernels is synchronized by barriers on the GPU they can only be executed on a single compute device.
                Therefore, using an async queue may increase the speed by allowing multiple kernels to execute in parallel (if local memory permits)
                */
                localWorkSize = halfReductionSize % optimalMultiples[KERNEL_sumInLocalMemoryCombined] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_sumInLocalMemoryCombined] - (halfReductionSize % optimalMultiples[KERNEL_sumInLocalMemoryCombined])), maximumElementsForLocalFPTcombinedSum);
                globalWorkSize = localWorkSize;
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemoryCombined]
                        .putArg(maskBuffer)
                        .putArg(entryImageBuffer)
                        .putArg(gradient0)
                        .putArg(gradient1)
                        .putArg(gradient2)
                        .putArg(hessian00)
                        .putArg(hessian01)
                        .putArg(hessian02)
                        .putArg(hessian11)
                        .putArg(hessian12)
                        .putArg(hessian22)
                        .putNullArg(localWorkSize*4 /*size in bytes of local mem allocation*/)
                        .putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemoryCombined].rewind();
                
                buffer = queue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                area = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(maskBuffer, buffer);
                buffer = queue.putMapBuffer(entryImageBuffer, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                mse = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(entryImageBuffer, buffer);
                buffer = queue.putMapBuffer(gradient0, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                gradient[0] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(gradient0, buffer);
                buffer = queue.putMapBuffer(gradient1, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                gradient[1] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(gradient1, buffer);
                buffer = queue.putMapBuffer(gradient2, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                gradient[2] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(gradient2, buffer);
                buffer = queue.putMapBuffer(hessian00, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[0][0] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian00, buffer);
                buffer = queue.putMapBuffer(hessian01, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[0][1] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian01, buffer);
                buffer = queue.putMapBuffer(hessian02, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[0][2] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian02, buffer);
                buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[1][1] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian11, buffer);
                buffer = queue.putMapBuffer(hessian12, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[1][2] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian12, buffer);
                buffer = queue.putMapBuffer(hessian22, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[2][2] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian22, buffer);
            }
            else
            {
                
                int nrOfBlocks = ((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))%blockSizesRigidBodyParallel;
                nrOfBlocks = nrOfBlocks == 0 ? (int)Math.min(((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))/blockSizesRigidBodyParallel,maximumSumReductionBlockNr) : (int)Math.min((((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))+(blockSizesRigidBodyParallel-nrOfBlocks))/blockSizesRigidBodyParallel,maximumSumReductionBlockNr);
                
                int globalWorkSize = nrOfBlocks * blockSizesRigidBodyParallel;
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent]
                        .putArg(sourcePyramid[pyramidIndex].Image)
                        .putArg(targetPyramid[pyramidIndex].Coefficient)
                        .putArg(sourcePyramid[pyramidIndex].xGradient)
                        .putArg(sourcePyramid[pyramidIndex].yGradient)
                        .putArg(gradient0)
                        .putArg(gradient1)
                        .putArg(gradient2)
                        .putArg(hessian00)
                        .putArg(hessian01)
                        .putArg(hessian02)
                        .putArg(hessian11)
                        .putArg(hessian12)
                        .putArg(hessian22)
                        .putArg(entryImageBuffer)
                        .putArg(maskBuffer);
                
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent]
                        .putNullArg(blockSizesRigidBodyParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesRigidBodyParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesRigidBodyParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesRigidBodyParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesRigidBodyParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesRigidBodyParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesRigidBodyParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesRigidBodyParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesRigidBodyParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesRigidBodyParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesRigidBodyParallel*4 /*size in bytes of local mem allocation*/);
                
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent]
                        .putArg((int)sourcePyramid[pyramidIndex].width)
                        .putArg((int)sourcePyramid[pyramidIndex].height)
                        .putArg((int)targetPyramid[pyramidIndex].width)
                        .putArg((int)targetPyramid[pyramidIndex].height);

                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent]
                        .putArg((float)currentoffsetx)
                        .putArg((float)curentoffsety)
                        .putArg((float)currentangle);
               
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent]
                        .putArg((int)targetPyramid[pyramidIndex].width*2)
                        .putArg((int)targetPyramid[pyramidIndex].height*2);
                
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent],0,globalWorkSize,blockSizesRigidBodyParallel);
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent].rewind();
                
                // use in-memory summations (also async) each with its own buffer to do the final reduction)
                halfReductionSize = (int) ((nrOfBlocks+(nrOfBlocks%2))/2);
                int localWorkSize = halfReductionSize % optimalMultiples[KERNEL_sumInLocalMemoryCombined] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_sumInLocalMemoryCombined] - (halfReductionSize % optimalMultiples[KERNEL_sumInLocalMemoryCombined])), maximumElementsForLocalFPTcombinedSum);
                globalWorkSize = localWorkSize;
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemoryCombined]
                        .putArg(gradient0)
                        .putArg(gradient1)
                        .putArg(gradient2)
                        .putArg(hessian00)
                        .putArg(hessian01)
                        .putArg(hessian02)
                        .putArg(hessian11)
                        .putArg(hessian12)
                        .putArg(hessian22)
                        .putArg(entryImageBuffer)
                        .putArg(maskBuffer)
                        .putNullArg(halfReductionSize*4 /*size in bytes of local mem allocation*/)
                        .putArg((int)(nrOfBlocks));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemoryCombined].rewind();
                
                buffer = queue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                area = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(maskBuffer, buffer);
                
                buffer = queue.putMapBuffer(entryImageBuffer, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                mse = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(entryImageBuffer, buffer);
                        
                buffer = queue.putMapBuffer(gradient0, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                gradient[0] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(gradient0, buffer);
                        
                buffer = queue.putMapBuffer(gradient1, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                gradient[1] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(gradient1, buffer);
                        
                buffer = queue.putMapBuffer(gradient2, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                gradient[2] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(gradient2, buffer);
                        
                buffer = queue.putMapBuffer(hessian00, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[0][0] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian00, buffer);
                        
                buffer = queue.putMapBuffer(hessian01, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[0][1] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian01, buffer);
                        
                buffer = queue.putMapBuffer(hessian02, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[0][2] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian02, buffer);
                        
                buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[1][1] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian11, buffer);
                        
                buffer = queue.putMapBuffer(hessian12, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[1][2] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian12, buffer);
                        
                buffer = queue.putMapBuffer(hessian22, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[2][2] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian22, buffer);
            }
            //symmetrize hessian
            for (int i = 1; (i < 3); i++) {
                for (int j = 0; (j < i); j++) {
                        hessian[i][j] = hessian[j][i];
                }
            }
            return mse/area;
        }
        
        private double getTranslationMeanSquares(int pyramidIndex, double currentoffsetx, double curentoffsety)
        {
            // now we have the diffs let's calculate all the derived elements
            int halfReductionSize = (int) (((sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)+((sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)%2))/2);
            ByteBuffer buffer;
            double area;
            // calculate the MSE row-wise
            double mse;
            if(halfReductionSize <= maximumElementsForLocalFPTcombinedSum)
            {                
                uniformBSplineTransformProgramKernels[KERNEL_ftranslationErrorWithGradAndHess]
                        .putArg(sourcePyramid[pyramidIndex].Image)
                        .putArg(targetPyramid[pyramidIndex].Coefficient)
                        .putArg(sourcePyramid[pyramidIndex].xGradient)
                        .putArg(sourcePyramid[pyramidIndex].yGradient)
                        .putArg(gradient0)
                        .putArg(gradient1)
                        .putArg(hessian00)
                        .putArg(hessian01)
                        .putArg(hessian11)
                        .putArg(entryImageBuffer)
                        .putArg(maskBuffer)
                        .putArg((int)sourcePyramid[pyramidIndex].width)
                        .putArg((int)sourcePyramid[pyramidIndex].height)
                        .putArg((int)targetPyramid[pyramidIndex].width)
                        .putArg((int)targetPyramid[pyramidIndex].height)
                        .putArg((float)currentoffsetx)
                        .putArg((float)curentoffsety);

                int localWorkSize = (int)Math.min(uniformBSplineTransformProgramKernels[KERNEL_ftranslationErrorWithGradAndHess].getWorkGroupSize(device),blocksizeMultiplier*optimalMultiples[KERNEL_ftranslationErrorWithGradAndHess]);  // Local work size dimensions
                int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_ftranslationErrorWithGradAndHess], (int)(sourcePyramid[pyramidIndex].width*sourcePyramid[pyramidIndex].height));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_ftranslationErrorWithGradAndHess],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_ftranslationErrorWithGradAndHess].rewind();
                
                /*
                Because each of the following kernels is synchronized by barriers on the GPU they can only be executed on a single compute device.
                Therefore using an async queue may increase the speed by allowing multiple kernels to execute in parallel (if local memory permits)
                */
                localWorkSize = halfReductionSize % optimalMultiples[KERNEL_ftranslationSumInLocalMemoryCombined] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_ftranslationSumInLocalMemoryCombined] - (halfReductionSize % optimalMultiples[KERNEL_ftranslationSumInLocalMemoryCombined])), maximumElementsForLocalFPTcombinedSum);
                globalWorkSize = localWorkSize;
                uniformBSplineTransformProgramKernels[KERNEL_ftranslationSumInLocalMemoryCombined]
                        .putArg(maskBuffer)
                        .putArg(entryImageBuffer)
                        .putArg(gradient0)
                        .putArg(gradient1)
                        .putArg(hessian00)
                        .putArg(hessian01)
                        .putArg(hessian11)
                        .putNullArg(localWorkSize*4 /*size in bytes of local mem allocation*/)
                        .putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_ftranslationSumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemoryCombined].rewind();
                
                buffer = queue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                area = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(maskBuffer, buffer);
                buffer = queue.putMapBuffer(entryImageBuffer, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                mse = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(entryImageBuffer, buffer);
                buffer = queue.putMapBuffer(gradient0, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                gradient[0] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(gradient0, buffer);
                buffer = queue.putMapBuffer(gradient1, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                gradient[1] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(gradient1, buffer);
                buffer = queue.putMapBuffer(hessian00, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[0][0] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian00, buffer);
                buffer = queue.putMapBuffer(hessian01, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[0][1] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian01, buffer);
                buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[1][1] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian11, buffer);
            }
            else
            {
                
                int nrOfBlocks = ((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))%blockSizesRigidBodyParallel;
                nrOfBlocks = nrOfBlocks == 0 ? (int)Math.min(((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))/blockSizesRigidBodyParallel,maximumSumReductionBlockNr) : (int)Math.min((((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))+(blockSizesRigidBodyParallel-nrOfBlocks))/blockSizesRigidBodyParallel,maximumSumReductionBlockNr);
                
                int globalWorkSize = nrOfBlocks * blockSizesRigidBodyParallel;
                uniformBSplineTransformProgramKernels[KERNEL_ftranslationErrorWithGradAndHessBrent]
                        .putArg(sourcePyramid[pyramidIndex].Image)
                        .putArg(targetPyramid[pyramidIndex].Coefficient)
                        .putArg(sourcePyramid[pyramidIndex].xGradient)
                        .putArg(sourcePyramid[pyramidIndex].yGradient)
                        .putArg(gradient0)
                        .putArg(gradient1)
                        .putArg(hessian00)
                        .putArg(hessian01)
                        .putArg(hessian11)
                        .putArg(entryImageBuffer)
                        .putArg(maskBuffer);
                
                uniformBSplineTransformProgramKernels[KERNEL_ftranslationErrorWithGradAndHessBrent]
                        .putNullArg(blockSizesRigidBodyParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesRigidBodyParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesRigidBodyParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesRigidBodyParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesRigidBodyParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesRigidBodyParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesRigidBodyParallel*4 /*size in bytes of local mem allocation*/);
                
                uniformBSplineTransformProgramKernels[KERNEL_ftranslationErrorWithGradAndHessBrent]
                        .putArg((int)sourcePyramid[pyramidIndex].width)
                        .putArg((int)sourcePyramid[pyramidIndex].height)
                        .putArg((int)targetPyramid[pyramidIndex].width)
                        .putArg((int)targetPyramid[pyramidIndex].height);

                uniformBSplineTransformProgramKernels[KERNEL_ftranslationErrorWithGradAndHessBrent]
                        .putArg((float)currentoffsetx)
                        .putArg((float)curentoffsety);
               
                uniformBSplineTransformProgramKernels[KERNEL_ftranslationErrorWithGradAndHessBrent]
                        .putArg((int)targetPyramid[pyramidIndex].width*2)
                        .putArg((int)targetPyramid[pyramidIndex].height*2);
                
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_ftranslationErrorWithGradAndHessBrent],0,globalWorkSize,blockSizesRigidBodyParallel);
                uniformBSplineTransformProgramKernels[KERNEL_ftranslationErrorWithGradAndHessBrent].rewind();
                
                //now use the in memory summations (also async) each with its own buffer to do the final reduction)
                halfReductionSize = (int) ((nrOfBlocks+(nrOfBlocks%2))/2);
                int localWorkSize = halfReductionSize % optimalMultiples[KERNEL_ftranslationSumInLocalMemoryCombined] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_ftranslationSumInLocalMemoryCombined] - (halfReductionSize % optimalMultiples[KERNEL_ftranslationSumInLocalMemoryCombined])), maximumElementsForLocalFPTcombinedSum);
                globalWorkSize = localWorkSize;
                uniformBSplineTransformProgramKernels[KERNEL_ftranslationSumInLocalMemoryCombined]
                        .putArg(gradient0)
                        .putArg(gradient1)
                        .putArg(hessian00)
                        .putArg(hessian01)
                        .putArg(hessian11)
                        .putArg(entryImageBuffer)
                        .putArg(maskBuffer)
                        .putNullArg(halfReductionSize*4 /*size in bytes of local mem allocation*/)
                        .putArg((int)(nrOfBlocks));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_ftranslationSumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_ftranslationSumInLocalMemoryCombined].rewind();
                
                buffer = queue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                area = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(maskBuffer, buffer);
                
                buffer = queue.putMapBuffer(entryImageBuffer, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                mse = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(entryImageBuffer, buffer);
                        
                buffer = queue.putMapBuffer(gradient0, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                gradient[0] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(gradient0, buffer);
                        
                buffer = queue.putMapBuffer(gradient1, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                gradient[1] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(gradient1, buffer);
                        
                buffer = queue.putMapBuffer(hessian00, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[0][0] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian00, buffer);
                        
                buffer = queue.putMapBuffer(hessian01, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[0][1] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian01, buffer);
                        
                buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[1][1] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian11, buffer);
            }
            //symmetrize hessian
            for (int i = 1; (i < 2); i++) {
                for (int j = 0; (j < i); j++) {
                        hessian[i][j] = hessian[j][i];
                }
            }
            return mse/area;
        }
        
        
        private double doubleGetRigidBodyMeanSquares(double currentoffsetx, double currentoffsety, double currentangle)
        {
            double area = 0.0;
            double mse = 0.0;
            if(!usesFloat)
            {
                // now we have the diffs let's calculate all the derived elements
                int halfReductionSize = (int) (((sourceDoubleSlice.width * sourceDoubleSlice.height)+((sourceDoubleSlice.width * sourceDoubleSlice.height)%2))/2);
                ByteBuffer buffer;
                // calculate the MSE row-wise
                if(halfReductionSize <= doublemaximumElementsForLocalFPTcombinedSum)
                {                
                    uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHess]
                            .putArg(sourceDoubleSlice.Image)
                            .putArg(targetDoubleSlice.Coefficient)
                            .putArg(sourceDoubleSlice.xGradient)
                            .putArg(sourceDoubleSlice.yGradient)
                            .putArg(gradient0)
                            .putArg(gradient1)
                            .putArg(gradient2)
                            .putArg(hessian00)
                            .putArg(hessian01)
                            .putArg(hessian02)
                            .putArg(hessian11)
                            .putArg(hessian12)
                            .putArg(hessian22)
                            .putArg(doubleEntryImageBuffer)
                            .putArg(maskBuffer)
                            .putArg((int)sourceDoubleSlice.width)
                            .putArg((int)sourceDoubleSlice.height)
                            .putArg((int)targetDoubleSlice.width)
                            .putArg((int)targetDoubleSlice.height)
                            .putArg(currentoffsetx)
                            .putArg(currentoffsety)
                            .putArg(currentangle);

                    int localWorkSize = (int)Math.min(uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHess].getWorkGroupSize(device),blocksizeMultiplier*optimalMultiples[KERNEL_drigidBodyErrorWithGradAndHess]);  // Local work size dimensions
                    int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_drigidBodyErrorWithGradAndHess], (int)(sourceDoubleSlice.width*sourceDoubleSlice.height));
                    queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHess],0,globalWorkSize,localWorkSize);
                    uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHess].rewind();

                    /*
                    Because each of the following kernels is synchronized by barriers on the GPU they can only be executed on a single compute device.
                    Therefore using an async queue may increase the speed by allowing multiple kernels to execute in parallel (if local memory permits)
                    */
                    localWorkSize = halfReductionSize % optimalMultiples[KERNEL_dsumInLocalMemoryCombined] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_dsumInLocalMemoryCombined] - (halfReductionSize % optimalMultiples[KERNEL_dsumInLocalMemoryCombined])), doublemaximumElementsForLocalFPTcombinedSum);
                    globalWorkSize = localWorkSize;
                    uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemoryCombined]
                            .putArg(maskBuffer)
                            .putArg(doubleEntryImageBuffer)
                            .putArg(gradient0)
                            .putArg(gradient1)
                            .putArg(gradient2)
                            .putArg(hessian00)
                            .putArg(hessian01)
                            .putArg(hessian02)
                            .putArg(hessian11)
                            .putArg(hessian12)
                            .putArg(hessian22)
                            .putNullArg(localWorkSize*8 /*size in bytes of local mem allocation*/)
                            .putArg((int)(sourceDoubleSlice.width * sourceDoubleSlice.height));
                    queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                    uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemoryCombined].rewind();

                    buffer = queue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    area = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(maskBuffer, buffer);
                    buffer = queue.putMapBuffer(doubleEntryImageBuffer, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    mse = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(doubleEntryImageBuffer, buffer);
                    buffer = queue.putMapBuffer(gradient0, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[0] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient0, buffer);
                    buffer = queue.putMapBuffer(gradient1, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[1] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient1, buffer);
                    buffer = queue.putMapBuffer(gradient2, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[2] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient2, buffer);
                    buffer = queue.putMapBuffer(hessian00, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][0] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian00, buffer);
                    buffer = queue.putMapBuffer(hessian01, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][1] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian01, buffer);
                    buffer = queue.putMapBuffer(hessian02, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][2] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian02, buffer);
                    buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[1][1] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian11, buffer);
                    buffer = queue.putMapBuffer(hessian12, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[1][2] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian12, buffer);
                    buffer = queue.putMapBuffer(hessian22, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[2][2] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian22, buffer);
                }
                else
                {

                    int nrOfBlocks = ((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))%doubleblockSizesRigidBodyParallel;
                    nrOfBlocks = nrOfBlocks == 0 ? (int)Math.min(((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))/doubleblockSizesRigidBodyParallel,maximumSumReductionBlockNr) : (int)Math.min((((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))+(doubleblockSizesRigidBodyParallel-nrOfBlocks))/doubleblockSizesRigidBodyParallel,maximumSumReductionBlockNr);

                    int globalWorkSize = nrOfBlocks * doubleblockSizesRigidBodyParallel;
                    uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHessBrent]
                            .putArg(sourceDoubleSlice.Image)
                            .putArg(targetDoubleSlice.Coefficient)
                            .putArg(sourceDoubleSlice.xGradient)
                            .putArg(sourceDoubleSlice.yGradient)
                            .putArg(gradient0)
                            .putArg(gradient1)
                            .putArg(gradient2)
                            .putArg(hessian00)
                            .putArg(hessian01)
                            .putArg(hessian02)
                            .putArg(hessian11)
                            .putArg(hessian12)
                            .putArg(hessian22)
                            .putArg(doubleEntryImageBuffer)
                            .putArg(maskBuffer);


                    uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHessBrent]
                            .putNullArg(blockSizesRigidBodyParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(blockSizesRigidBodyParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(blockSizesRigidBodyParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(blockSizesRigidBodyParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(blockSizesRigidBodyParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(blockSizesRigidBodyParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(blockSizesRigidBodyParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(blockSizesRigidBodyParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(blockSizesRigidBodyParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(blockSizesRigidBodyParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(blockSizesRigidBodyParallel*8 /*size in bytes of local mem allocation*/);

                    uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHessBrent]
                            .putArg((int)sourceDoubleSlice.width)
                            .putArg((int)sourceDoubleSlice.height)
                            .putArg((int)targetDoubleSlice.width)
                            .putArg((int)targetDoubleSlice.height);

                    uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHessBrent]
                            .putArg(currentoffsetx)
                            .putArg(currentoffsety)
                            .putArg(currentangle);

                    uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHessBrent]
                            .putArg((int)targetDoubleSlice.width*2)
                            .putArg((int)targetDoubleSlice.height*2);                
                    queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHessBrent],0,globalWorkSize,doubleblockSizesRigidBodyParallel);
                    uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHessBrent].rewind();

                    // use in-memory summations (also async) each with its own buffer to do the final reduction)
                    halfReductionSize = (int) ((nrOfBlocks+(nrOfBlocks%2))/2);
                    int localWorkSize = halfReductionSize % optimalMultiples[KERNEL_dsumInLocalMemoryCombined] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_dsumInLocalMemoryCombined] - (halfReductionSize % optimalMultiples[KERNEL_dsumInLocalMemoryCombined])), doublemaximumElementsForLocalFPTcombinedSum);
                    globalWorkSize = localWorkSize;
                    uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemoryCombined]
                            .putArg(gradient0)
                            .putArg(gradient1)
                            .putArg(gradient2)
                            .putArg(hessian00)
                            .putArg(hessian01)
                            .putArg(hessian02)
                            .putArg(hessian11)
                            .putArg(hessian12)
                            .putArg(hessian22)
                            .putArg(doubleEntryImageBuffer)
                            .putArg(maskBuffer)
                            .putNullArg(halfReductionSize*8 /*size in bytes of local mem allocation*/)
                            .putArg((int)(nrOfBlocks));
                    queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                    uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemoryCombined].rewind();

                    buffer = queue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    area = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(maskBuffer, buffer);

                    buffer = queue.putMapBuffer(doubleEntryImageBuffer, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    mse = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(doubleEntryImageBuffer, buffer);

                    buffer = queue.putMapBuffer(gradient0, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[0] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient0, buffer);

                    buffer = queue.putMapBuffer(gradient1, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[1] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient1, buffer);

                    buffer = queue.putMapBuffer(gradient2, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[2] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient2, buffer);

                    buffer = queue.putMapBuffer(hessian00, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][0] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian00, buffer);

                    buffer = queue.putMapBuffer(hessian01, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][1] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian01, buffer);

                    buffer = queue.putMapBuffer(hessian02, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][2] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian02, buffer);

                    buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[1][1] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian11, buffer);

                    buffer = queue.putMapBuffer(hessian12, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[1][2] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian12, buffer);

                    buffer = queue.putMapBuffer(hessian22, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[2][2] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian22, buffer);
                }
            }
            else
            {
                // First reset the global values which will not be reset in the loop
                gradient[0] = 0.0;
                gradient[1] = 0.0;
                gradient[2] = 0.0;
                hessian[0][0] = 0.0;
                hessian[0][1] = 0.0;
                hessian[0][2] = 0.0;
                hessian[1][1] = 0.0;
                hessian[1][2] = 0.0;
                hessian[2][2] = 0.0;
                final int width = (int)sharedContext.img.dimension(0);
                final int height = (int)sharedContext.img.dimension(1);
                final int doubletargetwidth = width * 2;
                final int doubletargetheight = height * 2;
                int nIndex = 0;
                int larea = 0;
                int p;
                int q;
                int tmpindex;
                double s;
                double xvecx = Math.cos(currentangle);
                double xvecy = -Math.sin(currentangle);
                double yvecx = -xvecy;
                double yvecy = xvecx;
                double coordx;
                double rescoordx;
                double coordy;
                double rescoordy;
                int mskx;
                int msky;
                for(int i = 0;i < height;i++)
                {
                	// First walk along the Y-vector direction and reset the X-position (otherwise the
                    // y position is initially correct and then lagging behind by one all the time)
                    coordx = currentoffsetx + ((double)i) * yvecx;
                    coordy = currentoffsety + ((double)i) * yvecy;
                    for(int n = 0;n < width;n++,nIndex++)
                    {
                        mskx = (int)Math.round(coordx);
                        msky = (int)Math.round(coordy);
                        if((mskx >= 0)&&(mskx < width)&&(msky >= 0)&&(msky < height))
                        {
                            larea++;
                            // Calculate X-interpolation indices
                            p = (coordx >= 0) ? (((int)coordx) + 2) : (((int)coordx) + 1);
                            for(int c = 0;c < 4;c++,p--)
                            {
                                q = (p < 0) ? (-1 - p) : p;
                                q = (q < doubletargetwidth) ? q : q % doubletargetwidth;
                                xInterpolationIndices[c] = q >= width ? (doubletargetwidth - 1 - q) : q;
                            }
                            // calculate Y-interpolation indices
                            p = (coordy >= 0) ? (((int)coordy) + 2) : (((int)coordy) + 1);
                            for(int c = 0;c < 4;c++,p--)
                            {
                                q = (p < 0) ? (-1 - p) : p;
                                q = (q < doubletargetheight) ? q : q % doubletargetheight;
                                yInterpolationIndices[c] = q >= height ? (doubletargetheight - 1 - q) * width : q * width;// calculate linearized coordinates of the coefficient array
                            }
                            // get the residuals of the coordinates
                            rescoordx = coordx - (coordx >= 0.0 ? ((double)((int)coordx)) : ((double)(((int)coordx) - 1)));
                            rescoordy = coordy - (coordy >= 0.0 ? ((double)((int)coordy)) : ((double)(((int)coordy) - 1)));
                            // calculate the X-weights
                            s = 1.0 - rescoordx;
                            xWeights[3] = Math.pow(s, 3.0) / 6.0;
                            s = rescoordx * rescoordx;
                            xWeights[2] = (2.0 / 3.0) - 0.5 * s * (2.0 - rescoordx);
                            xWeights[0] = s * rescoordx / 6.0;
                            xWeights[1] = 1.0 - xWeights[0] - xWeights[2] - xWeights[3];
                            // calculate the Y-weights
                            s = 1.0 - rescoordy;
                            yWeights[3] = Math.pow(s, 3.0) / 6.0;
                            s = rescoordy * rescoordy;
                            yWeights[2] = (2.0 / 3.0) - 0.5 * s * (2.0 - rescoordy);
                            yWeights[0] = s * rescoordy / 6.0;
                            yWeights[1] = 1.0 - yWeights[0] - yWeights[2] - yWeights[3];

                            // calculate the interpolated value from the target coefficients
                            s = 0.0;
                            for(int y = 0;y < 4;y++)
                            {
                                rescoordx = 0.0;// To avoid using too many variables this one will be repurposed
                                tmpindex = yInterpolationIndices[y];
                                for(int x = 0;x < 4; x++)
                                {
                                    rescoordx += xWeights[x] * targetCoefficientDoubleSlice[tmpindex + xInterpolationIndices[x]];
                                }
                                s += yWeights[y] * rescoordx;
                            }

                            // calculate the values for returning
                            rescoordx = sourceImageDoubleSlice[nIndex] - s;// repurposed for diff
                            mse += rescoordx * rescoordx;
                            rescoordy = sourceyGradientDoubleSlice[nIndex] * (double)n - sourcexGradientDoubleSlice[nIndex] * (double)i;//repurposed for Theta
                            /*
                            TODO/FIXME/KNOWN ISSUE:
                            The following summation is MUCH worse than the parallel sum reduction done on the GPU, because (relatively speaking)
                            small numbers are added to an ever growing larger number reducing the precision in the outcome. Currently
                            I ignore this like the original implementation, but this is one of many reasons why the GPU version and the CPU version
                            will never yield the same results!
                            */
                            gradient[0] += rescoordx * rescoordy;
                            gradient[1] += rescoordx * sourcexGradientDoubleSlice[nIndex];
                            gradient[2] += rescoordx * sourceyGradientDoubleSlice[nIndex];
                            hessian[0][0] += rescoordy * rescoordy;
                            hessian[0][1] += rescoordy * sourcexGradientDoubleSlice[nIndex];
                            hessian[0][2] += rescoordy * sourceyGradientDoubleSlice[nIndex];
                            hessian[1][1] += sourcexGradientDoubleSlice[nIndex] * sourcexGradientDoubleSlice[nIndex];
                            hessian[1][2] += sourcexGradientDoubleSlice[nIndex] * sourceyGradientDoubleSlice[nIndex];
                            hessian[2][2] += sourceyGradientDoubleSlice[nIndex] * sourceyGradientDoubleSlice[nIndex];
                        }
                        // walk along the X-vector direction
                        coordx += xvecx;
                        coordy += xvecy;
                    }
                }
                area = (double)larea;
            }
            // symmetrize hessian
            for (int i = 1; (i < 3); i++) {
                for (int j = 0; (j < i); j++) {
                        hessian[i][j] = hessian[j][i];
                }
            }
            return mse/area;
        }
        
        
        private double doubleGetTranslationMeanSquares(double currentoffsetx, double currentoffsety)
        {
            double area = 0.0;
            double mse = 0.0;
            if(!usesFloat)
            {
                // we have the diffs let's calculate all the derived elements
                int halfReductionSize = (int) (((sourceDoubleSlice.width * sourceDoubleSlice.height)+((sourceDoubleSlice.width * sourceDoubleSlice.height)%2))/2);
                ByteBuffer buffer;
                // calculate the MSE row-wise
                if(halfReductionSize <= doublemaximumElementsForLocalFPTcombinedSum)
                {                
                    uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHess]
                            .putArg(sourceDoubleSlice.Image)
                            .putArg(targetDoubleSlice.Coefficient)
                            .putArg(sourceDoubleSlice.xGradient)
                            .putArg(sourceDoubleSlice.yGradient)
                            .putArg(gradient0)
                            .putArg(gradient1)
                            .putArg(hessian00)
                            .putArg(hessian01)
                            .putArg(hessian11)
                            .putArg(doubleEntryImageBuffer)
                            .putArg(maskBuffer)
                            .putArg((int)sourceDoubleSlice.width)
                            .putArg((int)sourceDoubleSlice.height)
                            .putArg((int)targetDoubleSlice.width)
                            .putArg((int)targetDoubleSlice.height)
                            .putArg(currentoffsetx)
                            .putArg(currentoffsety);

                    int localWorkSize = (int)Math.min(uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHess].getWorkGroupSize(device),blocksizeMultiplier*optimalMultiples[KERNEL_dtranslationErrorWithGradAndHess]);  // Local work size dimensions
                    int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dtranslationErrorWithGradAndHess], (int)(sourceDoubleSlice.width*sourceDoubleSlice.height));
                    queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHess],0,globalWorkSize,localWorkSize);
                    uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHess].rewind();

                    /*
                    Because each of the following kernels is synchronized by barriers on the GPU they can only be executed on a single compute device.
                    Therefore using an async queue may increase the speed by allowing multiple kernels to execute in parallel (if local memory permits)
                    */
                    localWorkSize = halfReductionSize % optimalMultiples[KERNEL_dtranslationSumInLocalMemoryCombined] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_dtranslationSumInLocalMemoryCombined] - (halfReductionSize % optimalMultiples[KERNEL_dtranslationSumInLocalMemoryCombined])), doublemaximumElementsForLocalFPTcombinedSum);
                    globalWorkSize = localWorkSize;
                    uniformBSplineTransformProgramKernels[KERNEL_dtranslationSumInLocalMemoryCombined]
                            .putArg(maskBuffer)
                            .putArg(doubleEntryImageBuffer)
                            .putArg(gradient0)
                            .putArg(gradient1)
                            .putArg(hessian00)
                            .putArg(hessian01)
                            .putArg(hessian11)
                            .putNullArg(localWorkSize*8 /*size in bytes of local mem allocation*/)
                            .putArg((int)(sourceDoubleSlice.width * sourceDoubleSlice.height));
                    queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dtranslationSumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                    uniformBSplineTransformProgramKernels[KERNEL_dtranslationSumInLocalMemoryCombined].rewind();

                    buffer = queue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    area = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(maskBuffer, buffer);
                    buffer = queue.putMapBuffer(doubleEntryImageBuffer, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    mse = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(doubleEntryImageBuffer, buffer);
                    buffer = queue.putMapBuffer(gradient0, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[0] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient0, buffer);
                    buffer = queue.putMapBuffer(gradient1, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[1] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient1, buffer);
                    buffer = queue.putMapBuffer(hessian00, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][0] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian00, buffer);
                    buffer = queue.putMapBuffer(hessian01, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][1] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian01, buffer);
                    buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[1][1] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian11, buffer);
                }
                else
                {

                    int nrOfBlocks = ((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))%doubleblockSizesRigidBodyParallel;
                    nrOfBlocks = nrOfBlocks == 0 ? (int)Math.min(((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))/doubleblockSizesRigidBodyParallel,maximumSumReductionBlockNr) : (int)Math.min((((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))+(doubleblockSizesRigidBodyParallel-nrOfBlocks))/doubleblockSizesRigidBodyParallel,maximumSumReductionBlockNr);

                    int globalWorkSize = nrOfBlocks * doubleblockSizesRigidBodyParallel;
                    uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHessBrent]
                            .putArg(sourceDoubleSlice.Image)
                            .putArg(targetDoubleSlice.Coefficient)
                            .putArg(sourceDoubleSlice.xGradient)
                            .putArg(sourceDoubleSlice.yGradient)
                            .putArg(gradient0)
                            .putArg(gradient1)
                            .putArg(hessian00)
                            .putArg(hessian01)
                            .putArg(hessian11)
                            .putArg(doubleEntryImageBuffer)
                            .putArg(maskBuffer);


                    uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHessBrent]
                            .putNullArg(blockSizesRigidBodyParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(blockSizesRigidBodyParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(blockSizesRigidBodyParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(blockSizesRigidBodyParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(blockSizesRigidBodyParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(blockSizesRigidBodyParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(blockSizesRigidBodyParallel*8 /*size in bytes of local mem allocation*/);

                    uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHessBrent]
                            .putArg((int)sourceDoubleSlice.width)
                            .putArg((int)sourceDoubleSlice.height)
                            .putArg((int)targetDoubleSlice.width)
                            .putArg((int)targetDoubleSlice.height);

                    uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHessBrent]
                            .putArg(currentoffsetx)
                            .putArg(currentoffsety);

                    uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHessBrent]
                            .putArg((int)targetDoubleSlice.width*2)
                            .putArg((int)targetDoubleSlice.height*2);                
                    queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHessBrent],0,globalWorkSize,doubleblockSizesRigidBodyParallel);
                    uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHessBrent].rewind();

                    // use in-memory summations (also async) each with its own buffer to do the final reduction)
                    halfReductionSize = (int) ((nrOfBlocks+(nrOfBlocks%2))/2);
                    int localWorkSize = halfReductionSize % optimalMultiples[KERNEL_dtranslationSumInLocalMemoryCombined] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_dtranslationSumInLocalMemoryCombined] - (halfReductionSize % optimalMultiples[KERNEL_dtranslationSumInLocalMemoryCombined])), doublemaximumElementsForLocalFPTcombinedSum);
                    globalWorkSize = localWorkSize;
                    uniformBSplineTransformProgramKernels[KERNEL_dtranslationSumInLocalMemoryCombined]
                            .putArg(gradient0)
                            .putArg(gradient1)
                            .putArg(hessian00)
                            .putArg(hessian01)
                            .putArg(hessian11)
                            .putArg(doubleEntryImageBuffer)
                            .putArg(maskBuffer)
                            .putNullArg(halfReductionSize*8 /*size in bytes of local mem allocation*/)
                            .putArg((int)(nrOfBlocks));
                    queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dtranslationSumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                    uniformBSplineTransformProgramKernels[KERNEL_dtranslationSumInLocalMemoryCombined].rewind();

                    buffer = queue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    area = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(maskBuffer, buffer);

                    buffer = queue.putMapBuffer(doubleEntryImageBuffer, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    mse = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(doubleEntryImageBuffer, buffer);

                    buffer = queue.putMapBuffer(gradient0, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[0] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient0, buffer);

                    buffer = queue.putMapBuffer(gradient1, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[1] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient1, buffer);

                    buffer = queue.putMapBuffer(hessian00, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][0] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian00, buffer);

                    buffer = queue.putMapBuffer(hessian01, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][1] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian01, buffer);

                    buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[1][1] = buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian11, buffer);
                }
            }
            else
            {
                // First reset the global values which will not be reset in the loop
                gradient[0] = 0.0;
                gradient[1] = 0.0;
                hessian[0][0] = 0.0;
                hessian[0][1] = 0.0;
                hessian[1][1] = 0.0;
                final int width = (int)sharedContext.img.dimension(0);
                final int height = (int)sharedContext.img.dimension(1);
                final int doubletargetwidth = width * 2;
                final int doubletargetheight = height * 2;
                int nIndex = 0;
                int larea = 0;
                int p;
                int q;
                int tmpindex;
                double s;
                double coordx;
                double rescoordx;
                double coordy;
                double rescoordy;
                int mskx;
                int msky;
                for(int i = 0;i < height;i++)
                {
                	// First walk along the Y-vector direction and reset the X-position (otherwise the
                    // y position is initially correct and then lagging behind by one all the time)
                    coordx = currentoffsetx;
                    coordy = currentoffsety + ((double)i);
                    for(int n = 0;n < width;n++,nIndex++)
                    {
                        mskx = (int)Math.round(coordx);
                        msky = (int)Math.round(coordy);
                        if((mskx >= 0)&&(mskx < width)&&(msky >= 0)&&(msky < height))
                        {
                            larea++;
                            // Calculate X-interpolation indices
                            p = (coordx >= 0) ? (((int)coordx) + 2) : (((int)coordx) + 1);
                            for(int c = 0;c < 4;c++,p--)
                            {
                                q = (p < 0) ? (-1 - p) : p;
                                q = (q < doubletargetwidth) ? q : q % doubletargetwidth;
                                xInterpolationIndices[c] = q >= width ? (doubletargetwidth - 1 - q) : q;
                            }
                            // calculate Y-interpolation indices
                            p = (coordy >= 0) ? (((int)coordy) + 2) : (((int)coordy) + 1);
                            for(int c = 0;c < 4;c++,p--)
                            {
                                q = (p < 0) ? (-1 - p) : p;
                                q = (q < doubletargetheight) ? q : q % doubletargetheight;
                                yInterpolationIndices[c] = q >= height ? (doubletargetheight - 1 - q) * width : q * width;// calculate linearized coordinates of the coefficient array
                            }
                            // get the residuals of the coordinates
                            rescoordx = coordx - (coordx >= 0.0 ? ((double)((int)coordx)) : ((double)(((int)coordx) - 1)));
                            rescoordy = coordy - (coordy >= 0.0 ? ((double)((int)coordy)) : ((double)(((int)coordy) - 1)));
                            // calculate the X-weights
                            s = 1.0 - rescoordx;
                            xWeights[3] = Math.pow(s, 3.0) / 6.0;
                            s = rescoordx * rescoordx;
                            xWeights[2] = (2.0 / 3.0) - 0.5 * s * (2.0 - rescoordx);
                            xWeights[0] = s * rescoordx / 6.0;
                            xWeights[1] = 1.0 - xWeights[0] - xWeights[2] - xWeights[3];
                            // calculate the Y-weights
                            s = 1.0 - rescoordy;
                            yWeights[3] = Math.pow(s, 3.0) / 6.0;
                            s = rescoordy * rescoordy;
                            yWeights[2] = (2.0 / 3.0) - 0.5 * s * (2.0 - rescoordy);
                            yWeights[0] = s * rescoordy / 6.0;
                            yWeights[1] = 1.0 - yWeights[0] - yWeights[2] - yWeights[3];

                            // calculate the interpolated value from the target coefficients
                            s = 0.0;
                            for(int y = 0;y < 4;y++)
                            {
                                rescoordx = 0.0;// To avoid using too many variables this one will be repurposed
                                tmpindex = yInterpolationIndices[y];
                                for(int x = 0;x < 4; x++)
                                {
                                    rescoordx += xWeights[x] * targetCoefficientDoubleSlice[tmpindex + xInterpolationIndices[x]];
                                }
                                s += yWeights[y] * rescoordx;
                            }

                            // calculate the values for returning
                            rescoordx = sourceImageDoubleSlice[nIndex] - s;// repurposed for diff
                            mse += rescoordx * rescoordx;
                            /*
                            TODO/FIXME/KNOWN ISSUE:
                            The following summation is MUCH worse than the parallel sum reduction done on the GPU, because (relatively speaking)
                            small numbers are added to an ever growing larger number reducing the precision in the outcome. Currently
                            I ignore this like the original implementation, but this is one of many reasons why the GPU version and the CPU version
                            will never yield the same results!
                            */
                            gradient[0] += rescoordx * sourcexGradientDoubleSlice[nIndex];
                            gradient[1] += rescoordx * sourceyGradientDoubleSlice[nIndex];
                            hessian[0][0] += sourcexGradientDoubleSlice[nIndex] * sourcexGradientDoubleSlice[nIndex];
                            hessian[0][1] += sourcexGradientDoubleSlice[nIndex] * sourceyGradientDoubleSlice[nIndex];
                            hessian[1][1] += sourceyGradientDoubleSlice[nIndex] * sourceyGradientDoubleSlice[nIndex];
                        }
                        // walk along the X-vector direction
                        coordx += 1.0;
                    }
                }
                area = (double)larea;
            }
            // symmetrize hessian
            for (int i = 1; (i < 2); i++) {
                for (int j = 0; (j < i); j++) {
                        hessian[i][j] = hessian[j][i];
                }
            }
            return mse/area;
        }
        
        
        private double doubleGetRigidBodyMeanSquaresWithoutHessian(double currentoffsetx, double currentoffsety, double currentangle)
        {
            double area = 0.0;
            double mse = 0.0;
            if(!usesFloat)
            {
                uniformBSplineTransformProgramKernels[KERNEL_drigidBodyError]
                        .putArg(sourceDoubleSlice.Image)
                        .putArg(targetDoubleSlice.Coefficient)
                        .putArg(doubleEntryImageBuffer)
                        .putArg(maskBuffer)
                        .putArg((int)sourceDoubleSlice.width)
                        .putArg((int)sourceDoubleSlice.height)
                        .putArg((int)targetDoubleSlice.width)
                        .putArg((int)targetDoubleSlice.height)
                        .putArg(currentoffsetx)
                        .putArg(currentoffsety)
                        .putArg(currentangle);          
                int localWorkSize = (int)Math.min(uniformBSplineTransformProgramKernels[KERNEL_drigidBodyError].getWorkGroupSize(device),optimalMultiples[KERNEL_drigidBodyError]*blocksizeMultiplier);  // Local work size dimensions
                int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_drigidBodyError], (int)(sourceDoubleSlice.width*sourceDoubleSlice.height));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_drigidBodyError],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_drigidBodyError].rewind();
                //now we have the diffs let's calculate all the derived elements

                // to transfer the data to the async queue it has to be synchronized first
                queue.finish();

                // calculate the MSE row-wise
                int halfReductionSize = (int) (((sourceDoubleSlice.width * sourceDoubleSlice.height)+((sourceDoubleSlice.width * sourceDoubleSlice.height)%2))/2);
                int nrOfBlocks;
                ByteBuffer buffer;

                if(halfReductionSize <= doublemaximumElementsForLocalFPTsum)
                {
                    /*
                    Because each of the following kernels is synchronized by barriers on the GPU they can only be executed on a single compute device.
                    Therefore using an async queue may increase the speed by allowing multiple kernels to execute in parallel (if local memory permits)
                    */
                    localWorkSize = halfReductionSize % optimalMultiples[KERNEL_dsumInLocalMemory] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_dsumInLocalMemory] - (halfReductionSize % optimalMultiples[KERNEL_dsumInLocalMemory])), doublemaximumElementsForLocalFPTsum);
                    globalWorkSize = localWorkSize;

                    uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory].putArg(maskBuffer).putNullArg(halfReductionSize*8 /*size in bytes of local mem allocation*/).putArg((int)(sourceDoubleSlice.width * sourceDoubleSlice.height));
                    asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory],0,globalWorkSize,localWorkSize);
                    uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory].rewind();
                    uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory].putArg(doubleEntryImageBuffer).putNullArg(halfReductionSize*8 /*size in bytes of local mem allocation*/).putArg((int)(sourceDoubleSlice.width * sourceDoubleSlice.height));
                    asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory],0,globalWorkSize,localWorkSize);
                    uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory].rewind();
                    asyncQueue.finish();
                    buffer = asyncQueue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    area = (double)buffer.asFloatBuffer().get();
                    asyncQueue.putUnmapMemory(maskBuffer, buffer);
                    buffer = asyncQueue.putMapBuffer(doubleEntryImageBuffer, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    mse = (double)buffer.asFloatBuffer().get();
                    asyncQueue.putUnmapMemory(doubleEntryImageBuffer, buffer);
                    asyncQueue.finish();
                }
                else
                {
                    nrOfBlocks = ((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))%doubleblockSizesFPT;
                    nrOfBlocks = nrOfBlocks == 0 ? (int)Math.min(((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))/doubleblockSizesFPT,maximumSumReductionBlockNr) : (int)Math.min((((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))+(doubleblockSizesFPT-nrOfBlocks))/doubleblockSizesFPT,maximumSumReductionBlockNr);

                    globalWorkSize = nrOfBlocks * doubleblockSizesFPT;
                    uniformBSplineTransformProgramKernels[KERNEL_dparallelGroupedSumReduction].putArg(maskBuffer).putArg(parallelSumReductionBuffers[0]).putArg((int)(sourceDoubleSlice.width * sourceDoubleSlice.height)).putNullArg(doubleblockSizesFPT*8 /*size in bytes of local mem allocation*/);
                    asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dparallelGroupedSumReduction],0,globalWorkSize,doubleblockSizesFPT);
                    uniformBSplineTransformProgramKernels[KERNEL_dparallelGroupedSumReduction].rewind();

                    uniformBSplineTransformProgramKernels[KERNEL_dparallelGroupedSumReduction].putArg(doubleEntryImageBuffer).putArg(parallelSumReductionBuffers[1]).putArg((int)(sourceDoubleSlice.width * sourceDoubleSlice.height)).putNullArg(doubleblockSizesFPT*8 /*size in bytes of local mem allocation*/);
                    asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dparallelGroupedSumReduction],0,globalWorkSize,doubleblockSizesFPT);
                    uniformBSplineTransformProgramKernels[KERNEL_dparallelGroupedSumReduction].rewind();

                    // synchronize the data
                    asyncQueue.finish();

                    // reduce in local memory
                    halfReductionSize = (int) ((nrOfBlocks+(nrOfBlocks%2))/2);
                    localWorkSize = halfReductionSize % optimalMultiples[KERNEL_dsumInLocalMemory] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_dsumInLocalMemory] - (halfReductionSize % optimalMultiples[KERNEL_dsumInLocalMemory])), doublemaximumElementsForLocalFPTsum);
                    globalWorkSize = localWorkSize;
                    uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory].putArg(parallelSumReductionBuffers[0]).putNullArg(halfReductionSize*8 /*size in bytes of local mem allocation*/).putArg(nrOfBlocks);
                    asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory],0,globalWorkSize,localWorkSize);
                    uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory].rewind();

                    uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory].putArg(parallelSumReductionBuffers[1]).putNullArg(halfReductionSize*8 /*size in bytes of local mem allocation*/).putArg(nrOfBlocks);
                    asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory],0,globalWorkSize,localWorkSize);
                    uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory].rewind();

                    // synchronize again
                    asyncQueue.finish();

                    // Download data
                    buffer = asyncQueue.putMapBuffer(parallelSumReductionBuffers[0], CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    area = (double)buffer.asFloatBuffer().get();
                    asyncQueue.putUnmapMemory(parallelSumReductionBuffers[0], buffer);
                    buffer = asyncQueue.putMapBuffer(parallelSumReductionBuffers[1], CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    mse = (double)buffer.asFloatBuffer().get();
                    asyncQueue.putUnmapMemory(parallelSumReductionBuffers[1], buffer);
                    asyncQueue.finish();
                }
            }
            else
            {
                final int width = (int)sharedContext.img.dimension(0);
                final int height = (int)sharedContext.img.dimension(1);
                final int doubletargetwidth = width * 2;
                final int doubletargetheight = height * 2;
                int nIndex = 0;
                int larea = 0;
                int p;
                int q;
                int tmpindex;
                double s;
                double xvecx = Math.cos(currentangle);
                double xvecy = -Math.sin(currentangle);
                double yvecx = -xvecy;
                double yvecy = xvecx;
                double coordx;
                double rescoordx;
                double coordy;
                double rescoordy;
                int mskx;
                int msky;
                for(int i = 0;i < height;i++)
                {
                	// First walk along the Y-vector direction and reset the X-position (otherwise the
                    // y position is initially correct and then lagging behind by one all the time)
                    coordx = currentoffsetx + ((double)i) * yvecx;
                    coordy = currentoffsety + ((double)i) * yvecy;
                    for(int n = 0;n < width;n++,nIndex++)
                    {
                        mskx = (int)Math.round(coordx);
                        msky = (int)Math.round(coordy);
                        if((mskx >= 0)&&(mskx < width)&&(msky >= 0)&&(msky < height))
                        {
                            larea++;
                            // Calculate X-interpolation indices
                            p = (coordx >= 0) ? (((int)coordx) + 2) : (((int)coordx) + 1);
                            for(int c = 0;c < 4;c++,p--)
                            {
                                q = (p < 0) ? (-1 - p) : p;
                                q = q < doubletargetwidth ? q : q % doubletargetwidth;
                                xInterpolationIndices[c] = q >= width ? (doubletargetwidth - 1 - q) : q;
                            }
                            // calculate Y-interpolation indices
                            p = (coordy >= 0) ? (((int)coordy) + 2) : (((int)coordy) + 1);
                            for(int c = 0;c < 4;c++,p--)
                            {
                                q = (p < 0) ? (-1 - p) : p;
                                q = q < doubletargetheight ? q : q % doubletargetheight;
                                yInterpolationIndices[c] = q >= height ? (doubletargetheight - 1 - q) * width : q * width;// calculate linearized coordinates of the coefficient array
                            }
                            // get the residuals of the coordinates
                            rescoordx = coordx - (coordx >= 0.0 ? ((double)((int)coordx)) : ((double)(((int)coordx) - 1)));
                            rescoordy = coordy - (coordy >= 0.0 ? ((double)((int)coordy)) : ((double)(((int)coordy) - 1)));
                            // calculate the X-weights
                            s = 1.0 - rescoordx;
                            xWeights[3] = Math.pow(s, 3.0) / 6.0;
                            s = rescoordx * rescoordx;
                            xWeights[2] = (2.0 / 3.0) - 0.5 * s * (2.0 - rescoordx);
                            xWeights[0] = s * rescoordx / 6.0;
                            xWeights[1] = 1.0 - xWeights[0] - xWeights[2] - xWeights[3];
                            // calculate the Y-weights
                            s = 1.0 - rescoordy;
                            yWeights[3] = Math.pow(s, 3.0) / 6.0;
                            s = rescoordy * rescoordy;
                            yWeights[2] = (2.0 / 3.0) - 0.5 * s * (2.0 - rescoordy);
                            yWeights[0] = s * rescoordy / 6.0;
                            yWeights[1] = 1.0 - yWeights[0] - yWeights[2] - yWeights[3];

                            // calculate the interpolated value from the target coefficients
                            s = 0.0;
                            for(int y = 0;y < 4;y++)
                            {
                                rescoordx = 0.0;// To avoid using too many variables this one will be repurposed
                                tmpindex = yInterpolationIndices[y];
                                for(int x = 0;x < 4; x++)
                                {
                                    rescoordx += xWeights[x] * targetCoefficientDoubleSlice[tmpindex + xInterpolationIndices[x]];
                                }
                                s += yWeights[y] * rescoordx;
                            }

                            // calculate the MSE
                            rescoordx = sourceImageDoubleSlice[nIndex] - s;// repurposed for diff
                            mse += rescoordx * rescoordx;
                        }
                        // walk along the X-vector direction
                        coordx += xvecx;
                        coordy += xvecy;
                    }
                }
                area = (double)larea;
            }
            return mse/area;
        }
        
        
        private double doubleGetTranslationMeanSquaresWithoutHessian(double currentoffsetx, double currentoffsety)
        {
            double area = 0.0;
            double mse = 0.0;
            if(!usesFloat)
            {
                uniformBSplineTransformProgramKernels[KERNEL_dtranslationError]
                        .putArg(sourceDoubleSlice.Image)
                        .putArg(targetDoubleSlice.Coefficient)
                        .putArg(doubleEntryImageBuffer)
                        .putArg(maskBuffer)
                        .putArg((int)sourceDoubleSlice.width)
                        .putArg((int)sourceDoubleSlice.height)
                        .putArg((int)targetDoubleSlice.width)
                        .putArg((int)targetDoubleSlice.height)
                        .putArg(currentoffsetx)
                        .putArg(currentoffsety);          
                int localWorkSize = (int)Math.min(uniformBSplineTransformProgramKernels[KERNEL_dtranslationError].getWorkGroupSize(device),optimalMultiples[KERNEL_dtranslationError]*blocksizeMultiplier);  // Local work size dimensions
                int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dtranslationError], (int)(sourceDoubleSlice.width*sourceDoubleSlice.height));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dtranslationError],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_dtranslationError].rewind();
                // now we have the diffs let's calculate all the derived elements

                // to transfer the data to the async queue it has to be synchronized first
                queue.finish();

                // calculate the MSE row-wise
                int halfReductionSize = (int) (((sourceDoubleSlice.width * sourceDoubleSlice.height)+((sourceDoubleSlice.width * sourceDoubleSlice.height)%2))/2);
                int nrOfBlocks;
                ByteBuffer buffer;

                if(halfReductionSize <= doublemaximumElementsForLocalFPTsum)
                {
                    /*
                    Because each of the following kernels is synchronized by barriers on the GPU they can only be executed on a single compute device.
                    Therefore using an async queue may increase the speed by allowing multiple kernels to execute in parallel (if local memory permits)
                    */
                    localWorkSize = halfReductionSize % optimalMultiples[KERNEL_dsumInLocalMemory] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_dsumInLocalMemory] - (halfReductionSize % optimalMultiples[KERNEL_dsumInLocalMemory])), doublemaximumElementsForLocalFPTsum);
                    globalWorkSize = localWorkSize;

                    uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory]
                            .putArg(maskBuffer)
                            .putNullArg(halfReductionSize*8 /*size in bytes of local mem allocation*/)
                            .putArg((int)(sourceDoubleSlice.width * sourceDoubleSlice.height));
                    asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory],0,globalWorkSize,localWorkSize);
                    uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory].rewind();
                    uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory]
                            .putArg(doubleEntryImageBuffer)
                            .putNullArg(halfReductionSize*8 /*size in bytes of local mem allocation*/)
                            .putArg((int)(sourceDoubleSlice.width * sourceDoubleSlice.height));
                    asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory],0,globalWorkSize,localWorkSize);
                    uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory].rewind();
                    asyncQueue.finish();
                    buffer = asyncQueue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    area = (double)buffer.asFloatBuffer().get();
                    asyncQueue.putUnmapMemory(maskBuffer, buffer);
                    buffer = asyncQueue.putMapBuffer(doubleEntryImageBuffer, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    mse = (double)buffer.asFloatBuffer().get();
                    asyncQueue.putUnmapMemory(doubleEntryImageBuffer, buffer);
                    asyncQueue.finish();
                }
                else
                {
                    nrOfBlocks = ((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))%doubleblockSizesFPT;
                    nrOfBlocks = nrOfBlocks == 0 ? (int)Math.min(((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))/doubleblockSizesFPT,maximumSumReductionBlockNr) : (int)Math.min((((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))+(doubleblockSizesFPT-nrOfBlocks))/doubleblockSizesFPT,maximumSumReductionBlockNr);

                    globalWorkSize = nrOfBlocks * doubleblockSizesFPT;
                    uniformBSplineTransformProgramKernels[KERNEL_dparallelGroupedSumReduction]
                            .putArg(maskBuffer)
                            .putArg(parallelSumReductionBuffers[0])
                            .putArg((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))
                            .putNullArg(doubleblockSizesFPT*8 /*size in bytes of local mem allocation*/);
                    asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dparallelGroupedSumReduction],0,globalWorkSize,doubleblockSizesFPT);
                    uniformBSplineTransformProgramKernels[KERNEL_dparallelGroupedSumReduction].rewind();

                    uniformBSplineTransformProgramKernels[KERNEL_dparallelGroupedSumReduction]
                            .putArg(doubleEntryImageBuffer)
                            .putArg(parallelSumReductionBuffers[1])
                            .putArg((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))
                            .putNullArg(doubleblockSizesFPT*8 /*size in bytes of local mem allocation*/);
                    asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dparallelGroupedSumReduction],0,globalWorkSize,doubleblockSizesFPT);
                    uniformBSplineTransformProgramKernels[KERNEL_dparallelGroupedSumReduction].rewind();

                    // synchronize the data
                    asyncQueue.finish();

                    // reduce in local memory
                    halfReductionSize = (int) ((nrOfBlocks+(nrOfBlocks%2))/2);
                    localWorkSize = halfReductionSize % optimalMultiples[KERNEL_dsumInLocalMemory] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_dsumInLocalMemory] - (halfReductionSize % optimalMultiples[KERNEL_dsumInLocalMemory])), doublemaximumElementsForLocalFPTsum);
                    globalWorkSize = localWorkSize;
                    uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory]
                            .putArg(parallelSumReductionBuffers[0])
                            .putNullArg(halfReductionSize*8 /*size in bytes of local mem allocation*/)
                            .putArg(nrOfBlocks);
                    asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory],0,globalWorkSize,localWorkSize);
                    uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory].rewind();

                    uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory]
                            .putArg(parallelSumReductionBuffers[1])
                            .putNullArg(halfReductionSize*8 /*size in bytes of local mem allocation*/)
                            .putArg(nrOfBlocks);
                    asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory],0,globalWorkSize,localWorkSize);
                    uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory].rewind();

                    // synchronize again
                    asyncQueue.finish();

                    // Download data
                    buffer = asyncQueue.putMapBuffer(parallelSumReductionBuffers[0], CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    area = (double)buffer.asFloatBuffer().get();
                    asyncQueue.putUnmapMemory(parallelSumReductionBuffers[0], buffer);
                    buffer = asyncQueue.putMapBuffer(parallelSumReductionBuffers[1], CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    mse = (double)buffer.asFloatBuffer().get();
                    asyncQueue.putUnmapMemory(parallelSumReductionBuffers[1], buffer);
                    asyncQueue.finish();
                }
            }
            else
            {
                final int width = (int)sharedContext.img.dimension(0);
                final int height = (int)sharedContext.img.dimension(1);
                final int doubletargetwidth = width * 2;
                final int doubletargetheight = height * 2;
                int nIndex = 0;
                int larea = 0;
                int p;
                int q;
                int tmpindex;
                double s;
                double coordx;
                double rescoordx;
                double coordy;
                double rescoordy;
                int mskx;
                int msky;
                for(int i = 0;i < height;i++)
                {
                	// First walk along the Y-vector direction and reset the X-position (otherwise the
                    // y position is initially correct and then lagging behind by one all the time)
                    coordx = currentoffsetx;
                    coordy = currentoffsety + ((double)i);
                    for(int n = 0;n < width;n++,nIndex++)
                    {
                        mskx = (int)Math.round(coordx);
                        msky = (int)Math.round(coordy);
                        if((mskx >= 0)&&(mskx < width)&&(msky >= 0)&&(msky < height))
                        {
                            larea++;
                            // Calculate X-interpolation indices
                            p = (coordx >= 0) ? (((int)coordx) + 2) : (((int)coordx) + 1);
                            for(int c = 0;c < 4;c++,p--)
                            {
                                q = (p < 0) ? (-1 - p) : p;
                                q = q < doubletargetwidth ? q : q % doubletargetwidth;
                                xInterpolationIndices[c] = q >= width ? (doubletargetwidth - 1 - q) : q;
                            }
                            // calculate Y-interpolation indices
                            p = (coordy >= 0) ? (((int)coordy) + 2) : (((int)coordy) + 1);
                            for(int c = 0;c < 4;c++,p--)
                            {
                                q = (p < 0) ? (-1 - p) : p;
                                q = q < doubletargetheight ? q : q % doubletargetheight;
                                yInterpolationIndices[c] = q >= height ? (doubletargetheight - 1 - q) * width : q * width;// calculate linearized coordinates of the coefficient array
                            }
                            // get the residuals of the coordinates
                            rescoordx = coordx - (coordx >= 0.0 ? ((double)((int)coordx)) : ((double)(((int)coordx) - 1)));
                            rescoordy = coordy - (coordy >= 0.0 ? ((double)((int)coordy)) : ((double)(((int)coordy) - 1)));
                            // calculate the X-weights
                            s = 1.0 - rescoordx;
                            xWeights[3] = Math.pow(s, 3.0) / 6.0;
                            s = rescoordx * rescoordx;
                            xWeights[2] = (2.0 / 3.0) - 0.5 * s * (2.0 - rescoordx);
                            xWeights[0] = s * rescoordx / 6.0;
                            xWeights[1] = 1.0 - xWeights[0] - xWeights[2] - xWeights[3];
                            // calculate the Y-weights
                            s = 1.0 - rescoordy;
                            yWeights[3] = Math.pow(s, 3.0) / 6.0;
                            s = rescoordy * rescoordy;
                            yWeights[2] = (2.0 / 3.0) - 0.5 * s * (2.0 - rescoordy);
                            yWeights[0] = s * rescoordy / 6.0;
                            yWeights[1] = 1.0 - yWeights[0] - yWeights[2] - yWeights[3];

                            // calculate the interpolated value from the target coefficients
                            s = 0.0;
                            for(int y = 0;y < 4;y++)
                            {
                                rescoordx = 0.0;// To avoid using too many variables this one will be repurposed
                                tmpindex = yInterpolationIndices[y];
                                for(int x = 0;x < 4; x++)
                                {
                                    rescoordx += xWeights[x] * targetCoefficientDoubleSlice[tmpindex + xInterpolationIndices[x]];
                                }
                                s += yWeights[y] * rescoordx;
                            }

                            // calculate the MSE
                            rescoordx = sourceImageDoubleSlice[nIndex] - s;// repurposed for diff
                            mse += rescoordx * rescoordx;
                        }
                        // walk along the X-vector direction
                        coordx += 1.0;
                    }
                }
                area = (double)larea;
            }
            return mse/area;
        }
        
        
        private double getRigidBodyMeanSquaresWithoutHessian(int pyramidIndex, double currentoffsetx, double curentoffsety, double currentangle)
        {
            uniformBSplineTransformProgramKernels[KERNEL_rigidBodyError]
                    .putArg(sourcePyramid[pyramidIndex].Image)
                    .putArg(targetPyramid[pyramidIndex].Coefficient)
                    .putArg(entryImageBuffer)
                    .putArg(maskBuffer)
                    .putArg((int)sourcePyramid[pyramidIndex].width)
                    .putArg((int)sourcePyramid[pyramidIndex].height)
                    .putArg((int)targetPyramid[pyramidIndex].width)
                    .putArg((int)targetPyramid[pyramidIndex].height)
                    .putArg((float)currentoffsetx)
                    .putArg((float)curentoffsety)
                    .putArg((float)currentangle);
            
            int localWorkSize = (int)Math.min(uniformBSplineTransformProgramKernels[KERNEL_rigidBodyError].getWorkGroupSize(device),optimalMultiples[KERNEL_rigidBodyError]*blocksizeMultiplier);  // Local work size dimensions
            int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_rigidBodyError], (int)(sourcePyramid[pyramidIndex].width*sourcePyramid[pyramidIndex].height));
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_rigidBodyError],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_rigidBodyError].rewind();
            // now we have the diffs let's calculate all the derived elements
            
            // to transfer the data to the async queue it has to be synchronized first
            queue.finish();
            
            // calculate the MSE row-wise
            int halfReductionSize = (int) (((sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)+((sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)%2))/2);
            int nrOfBlocks;
            ByteBuffer buffer;
            double area;
            double mse;
            if(halfReductionSize <= maximumElementsForLocalFPTsum)
            {
                /*
                Because each of the following kernels is synchronized by barriers on the GPU they can only be executed on a single compute device.
                Therefore using an async queue may increase the speed by allowing multiple kernels to execute in parallel (if local memory permits)
                */
                localWorkSize = halfReductionSize % optimalMultiples[KERNEL_sumInLocalMemory] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_sumInLocalMemory] - (halfReductionSize % optimalMultiples[KERNEL_sumInLocalMemory])), maximumElementsForLocalFPTsum);
                globalWorkSize = localWorkSize;
                
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].putArg(maskBuffer).putNullArg(halfReductionSize*4 /*size in bytes of local mem allocation*/).putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height));
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].putArg(entryImageBuffer).putNullArg(halfReductionSize*4 /*size in bytes of local mem allocation*/).putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height));
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                asyncQueue.finish();
                buffer = asyncQueue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                area = (double)buffer.asFloatBuffer().get();
                asyncQueue.putUnmapMemory(maskBuffer, buffer);
                buffer = asyncQueue.putMapBuffer(entryImageBuffer, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                mse = (double)buffer.asFloatBuffer().get();
                asyncQueue.putUnmapMemory(entryImageBuffer, buffer);
                asyncQueue.finish();
            }
            else
            {
                nrOfBlocks = ((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))%blockSizesFPT;
                nrOfBlocks = nrOfBlocks == 0 ? (int)Math.min(((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))/blockSizesFPT,maximumSumReductionBlockNr) : (int)Math.min((((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))+(blockSizesFPT-nrOfBlocks))/blockSizesFPT,maximumSumReductionBlockNr);
                
                globalWorkSize = nrOfBlocks * blockSizesFPT;
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].putArg(maskBuffer).putArg(parallelSumReductionBuffers[0]).putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)).putNullArg(blockSizesFPT*4 /*size in bytes of local mem allocation*/);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction],0,globalWorkSize,blockSizesFPT);
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].rewind();
                
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].putArg(entryImageBuffer).putArg(parallelSumReductionBuffers[1]).putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)).putNullArg(blockSizesFPT*4 /*size in bytes of local mem allocation*/);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction],0,globalWorkSize,blockSizesFPT);
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].rewind();
                
                // synchronize the data
                asyncQueue.finish();
                
                // reduce in local memory
                halfReductionSize = (int) ((nrOfBlocks+(nrOfBlocks%2))/2);
                localWorkSize = halfReductionSize % optimalMultiples[KERNEL_sumInLocalMemory] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_sumInLocalMemory] - (halfReductionSize % optimalMultiples[KERNEL_sumInLocalMemory])), maximumElementsForLocalFPTsum);
                globalWorkSize = localWorkSize;
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].putArg(parallelSumReductionBuffers[0]).putNullArg(halfReductionSize*4 /*size in bytes of local mem allocation*/).putArg(nrOfBlocks);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].putArg(parallelSumReductionBuffers[1]).putNullArg(halfReductionSize*4 /*size in bytes of local mem allocation*/).putArg(nrOfBlocks);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                
                // synchronize again
                asyncQueue.finish();
                
                // Download data
                buffer = asyncQueue.putMapBuffer(parallelSumReductionBuffers[0], CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                area = (double)buffer.asFloatBuffer().get();
                asyncQueue.putUnmapMemory(parallelSumReductionBuffers[0], buffer);
                buffer = asyncQueue.putMapBuffer(parallelSumReductionBuffers[1], CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                mse = (double)buffer.asFloatBuffer().get();
                asyncQueue.putUnmapMemory(parallelSumReductionBuffers[1], buffer);
                asyncQueue.finish();
            }
            return mse/area;
        }
        
        
        private double getTranslationMeanSquaresWithoutHessian(int pyramidIndex, double currentoffsetx, double curentoffsety)
        {
            uniformBSplineTransformProgramKernels[KERNEL_ftranslationError]
                    .putArg(sourcePyramid[pyramidIndex].Image)
                    .putArg(targetPyramid[pyramidIndex].Coefficient)
                    .putArg(entryImageBuffer)
                    .putArg(maskBuffer)
                    .putArg((int)sourcePyramid[pyramidIndex].width)
                    .putArg((int)sourcePyramid[pyramidIndex].height)
                    .putArg((int)targetPyramid[pyramidIndex].width)
                    .putArg((int)targetPyramid[pyramidIndex].height)
                    .putArg((float)currentoffsetx)
                    .putArg((float)curentoffsety);
            
            int localWorkSize = (int)Math.min(uniformBSplineTransformProgramKernels[KERNEL_ftranslationError].getWorkGroupSize(device),optimalMultiples[KERNEL_ftranslationError]*blocksizeMultiplier);  // Local work size dimensions
            int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_ftranslationError], (int)(sourcePyramid[pyramidIndex].width*sourcePyramid[pyramidIndex].height));
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_ftranslationError],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_ftranslationError].rewind();
            // now we have the diffs let's calculate all the derived elements
            
            // to transfer the data to the async queue it has to be synchronized first
            queue.finish();
            
            // calculate the MSE row-wise
            int halfReductionSize = (int) (((sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)+((sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)%2))/2);
            int nrOfBlocks;
            ByteBuffer buffer;
            double area;
            double mse;
            if(halfReductionSize <= maximumElementsForLocalFPTsum)
            {
                /*
                Because each of the following kernels is synchronized by barriers on the GPU they can only be executed on a single compute device.
                Therefore using an async queue may increase the speed by allowing multiple kernels to execute in parallel (if local memory permits)
                */
                localWorkSize = halfReductionSize % optimalMultiples[KERNEL_sumInLocalMemory] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_sumInLocalMemory] - (halfReductionSize % optimalMultiples[KERNEL_sumInLocalMemory])), maximumElementsForLocalFPTsum);
                globalWorkSize = localWorkSize;
                
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory]
                        .putArg(maskBuffer)
                        .putNullArg(halfReductionSize*4 /*size in bytes of local mem allocation*/)
                        .putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height));
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory]
                        .putArg(entryImageBuffer)
                        .putNullArg(halfReductionSize*4 /*size in bytes of local mem allocation*/)
                        .putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height));
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                asyncQueue.finish();
                buffer = asyncQueue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                area = (double)buffer.asFloatBuffer().get();
                asyncQueue.putUnmapMemory(maskBuffer, buffer);
                buffer = asyncQueue.putMapBuffer(entryImageBuffer, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                mse = (double)buffer.asFloatBuffer().get();
                asyncQueue.putUnmapMemory(entryImageBuffer, buffer);
                asyncQueue.finish();
            }
            else
            {
                nrOfBlocks = ((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))%blockSizesFPT;
                nrOfBlocks = nrOfBlocks == 0 ? (int)Math.min(((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))/blockSizesFPT,maximumSumReductionBlockNr) : (int)Math.min((((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))+(blockSizesFPT-nrOfBlocks))/blockSizesFPT,maximumSumReductionBlockNr);
                
                globalWorkSize = nrOfBlocks * blockSizesFPT;
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction]
                        .putArg(maskBuffer)
                        .putArg(parallelSumReductionBuffers[0])
                        .putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))
                        .putNullArg(blockSizesFPT*4 /*size in bytes of local mem allocation*/);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction],0,globalWorkSize,blockSizesFPT);
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].rewind();
                
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction]
                        .putArg(entryImageBuffer)
                        .putArg(parallelSumReductionBuffers[1])
                        .putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))
                        .putNullArg(blockSizesFPT*4 /*size in bytes of local mem allocation*/);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction],0,globalWorkSize,blockSizesFPT);
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].rewind();
                
                // synchronize the data
                asyncQueue.finish();
                
                // reduce in local memory
                halfReductionSize = (int) ((nrOfBlocks+(nrOfBlocks%2))/2);
                localWorkSize = halfReductionSize % optimalMultiples[KERNEL_sumInLocalMemory] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_sumInLocalMemory] - (halfReductionSize % optimalMultiples[KERNEL_sumInLocalMemory])), maximumElementsForLocalFPTsum);
                globalWorkSize = localWorkSize;
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory]
                        .putArg(parallelSumReductionBuffers[0])
                        .putNullArg(halfReductionSize*4 /*size in bytes of local mem allocation*/)
                        .putArg(nrOfBlocks);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory]
                        .putArg(parallelSumReductionBuffers[1])
                        .putNullArg(halfReductionSize*4 /*size in bytes of local mem allocation*/)
                        .putArg(nrOfBlocks);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                
                // synchronize again
                asyncQueue.finish();
                
                // Download data
                buffer = asyncQueue.putMapBuffer(parallelSumReductionBuffers[0], CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                area = (double)buffer.asFloatBuffer().get();
                asyncQueue.putUnmapMemory(parallelSumReductionBuffers[0], buffer);
                buffer = asyncQueue.putMapBuffer(parallelSumReductionBuffers[1], CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                mse = (double)buffer.asFloatBuffer().get();
                asyncQueue.putUnmapMemory(parallelSumReductionBuffers[1], buffer);
                asyncQueue.finish();
            }
            return mse/area;
        }
     
        private void constructSourceImagePyramid()
        {
            int width = (int)sharedContext.img.dimension(0);
            int height = (int)sharedContext.img.dimension(1);
            if(!usesFloat)
            {
                /*
                 * the image is in
                 * doubleEntryImageBuffer
                 * sourceDoubleSlice.Image
                 * (float) sourcePyramid[0].Image
                 * and the GPU may be used for double calculation 
                 * 
                */
                int localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2Dpremulhp].getWorkGroupSize(device);  // Local work size dimensions
                int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dCubicBSplinePrefilter2Dpremulhp], width*height);   // rounded up to the nearest multiple of the localWorkSize
                // pre-multiply the image
                uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2Dpremulhp].putArg(doubleEntryImageBuffer).putArg(width*height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2Dpremulhp],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DXhp].putArg(doubleEntryImageBuffer).putArg(width).putArg(height);
                // Conversion to B-spline coefficients along X axis (Group size must be >= height)
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DXhp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dCubicBSplinePrefilter2DXhp], height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DXhp],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DXhp].rewind();
                // X-derivatives
                uniformBSplineTransformProgramKernels[KERNEL_dantiSymmetricFirMirrorOffBounds1DX].putArg(doubleEntryImageBuffer).putArg(sourceDoubleSlice.xGradient).putArg(width).putArg(height);
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dantiSymmetricFirMirrorOffBounds1DX].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dantiSymmetricFirMirrorOffBounds1DX], width*height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dantiSymmetricFirMirrorOffBounds1DX],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_dantiSymmetricFirMirrorOffBounds1DX].rewind();
                // Now along the Y-axis
                // Has to be pre-multiplied by lambda again!!! (the kernel is still setup correctly)
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2Dpremulhp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dCubicBSplinePrefilter2Dpremulhp], width*height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2Dpremulhp],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2Dpremulhp].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DYhp].putArg(doubleEntryImageBuffer).putArg(width).putArg(height);
                // Conversion to B-spline coefficients along Y axis (Group size must be >= width)
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DYhp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dCubicBSplinePrefilter2DYhp], width);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DYhp],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DYhp].rewind();
                // No need to copy the B-spline coefficients for the source, because only the images are required and those were already copied during the conversion step

                // The Y-derivatives still need to calculate from the Y-coefficients
                // First calculate the Y-coefficients and only the Y-coefficients
                // Has to be pre-multiplied by lambda again!!! avoid copying data so use an out-of-place modifying calculation
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dTargetedCubicBSplinePrefilter2Dpremulhp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dTargetedCubicBSplinePrefilter2Dpremulhp], width*height);
                uniformBSplineTransformProgramKernels[KERNEL_dTargetedCubicBSplinePrefilter2Dpremulhp].putArg(sourceDoubleSlice.Image).putArg(doubleFullSizedGPUResidentHelperBuffer).putArg(width*height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dTargetedCubicBSplinePrefilter2Dpremulhp],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_dTargetedCubicBSplinePrefilter2Dpremulhp].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DYhp].putArg(doubleFullSizedGPUResidentHelperBuffer).putArg(width).putArg(height);
                // Conversion to B-spline coefficients along Y axis (Group size must be >= width)
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DYhp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dCubicBSplinePrefilter2DYhp], width);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DYhp],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DYhp].rewind();
                // Calculate the derivatives
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dantiSymmetricFirMirrorOffBounds1DY].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dantiSymmetricFirMirrorOffBounds1DY], width*height);
                uniformBSplineTransformProgramKernels[KERNEL_dantiSymmetricFirMirrorOffBounds1DY].putArg(doubleFullSizedGPUResidentHelperBuffer).putArg(sourceDoubleSlice.yGradient).putArg(width).putArg(height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dantiSymmetricFirMirrorOffBounds1DY],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_dantiSymmetricFirMirrorOffBounds1DY].rewind();
                
                /*
                    the coefficients are now in doubleEntryImageBuffer
                    the x gradient in sourceDoubleSlice.xGradient and
                    the y gradient in sourceDoubleSlice.yGradient
                    They only need to be converted and copied
                */
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_ConvertDoubleToFloat].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_ConvertDoubleToFloat], width*height);
                
                uniformBSplineTransformProgramKernels[KERNEL_ConvertDoubleToFloat].putArg(doubleEntryImageBuffer).putArg(entryImageBuffer).putArg(width*height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_ConvertDoubleToFloat],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_ConvertDoubleToFloat].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_ConvertDoubleToFloat].putArg(sourceDoubleSlice.xGradient).putArg(sourcePyramid[0].xGradient).putArg(width*height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_ConvertDoubleToFloat],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_ConvertDoubleToFloat].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_ConvertDoubleToFloat].putArg(sourceDoubleSlice.yGradient).putArg(sourcePyramid[0].yGradient).putArg(width*height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_ConvertDoubleToFloat],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_ConvertDoubleToFloat].rewind();
            }
            else
            {
                /*
                 * the image is in
                 * CPUentryImageBuffer
                 * sourceImageDoubleSlice
                 * (float) sourcePyramid[0].Image
                 * and double precision is not available 
                 * 
                */
                // Conversion to B-spline coefficients
                // pre-multiply the image for cubic spline interpolation
                PlainJavaCPUAligner.premultiplyCubicBSpline(CPUentryImageBuffer, width * height);
                // Conversion to B-spline coefficients along X axis
                PlainJavaCPUAligner.cubicBSplinePrefilter2DXhp(CPUentryImageBuffer, width, height);

                // X-derivatives
                PlainJavaCPUAligner.antiSymmetricFirMirrorOffBounds1DX(CPUentryImageBuffer, sourcexGradientDoubleSlice, width, height);

                // pre-multiply again
                PlainJavaCPUAligner.premultiplyCubicBSpline(CPUentryImageBuffer, width * height);
                // along the Y-axis
                PlainJavaCPUAligner.cubicBSplinePrefilter2DYhp(CPUentryImageBuffer, width, height);
                // for the source only  the images are needed in the pyramid so no need to copy the B-spline coefficients
                // convert to float and copy to the GPU
                ByteBuffer buffer = queue.putMapBuffer(entryImageBuffer, CLMemory.Map.WRITE, true);
                buffer.rewind();
                FloatBuffer fb = buffer.asFloatBuffer();
                for(int i = 0;i < CPUentryImageBuffer.length;i++)
                {
                    fb.put((float)CPUentryImageBuffer[i]);
                }
                queue.putUnmapMemory(entryImageBuffer, buffer); 
                //now the CPUentryImageBuffer can be overwritten

                // The Y-derivatives still need to be calculated from the Y-coefficients
                // First calculate the Y-coefficients and only the Y-coefficients
                // Has to be pre-multiplied by lambda again, avoid copying data so use an out-of-place modifying calculation
                PlainJavaCPUAligner.targetedPremultiplyCubicBSpline(sourceImageDoubleSlice, CPUentryImageBuffer, width*height);
                PlainJavaCPUAligner.cubicBSplinePrefilter2DYhp(CPUentryImageBuffer, width, height);
                // Now calculate the derivatives in Y-direction
                PlainJavaCPUAligner.antiSymmetricFirMirrorOffBounds1DY(CPUentryImageBuffer, sourceyGradientDoubleSlice, width, height);
                // convert and copy the gradient data
                buffer = queue.putMapBuffer(sourcePyramid[0].xGradient, CLMemory.Map.WRITE, true);
                buffer.rewind();
                fb = buffer.asFloatBuffer();
                for(int i = 0;i < sourcexGradientDoubleSlice.length;i++)
                {
                    fb.put((float)sourcexGradientDoubleSlice[i]);
                }
                queue.putUnmapMemory(sourcePyramid[0].xGradient, buffer);
                
                buffer = queue.putMapBuffer(sourcePyramid[0].yGradient, CLMemory.Map.WRITE, true);
                buffer.rewind();
                fb = buffer.asFloatBuffer();
                for(int i = 0;i < sourceyGradientDoubleSlice.length;i++)
                {
                    fb.put((float)sourceyGradientDoubleSlice[i]);
                }
                queue.putUnmapMemory(sourcePyramid[0].yGradient, buffer);
            }
            // TODO: use localWorkGroup size in a more sensible manner
            // Prepare the image for resampling by applying the FIR filter of degree 7 (out of place mod)
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhpDeg7].putArg(entryImageBuffer).putArg(fullSizedGPUResidentHelperBuffer).putArg(width).putArg(height);
            int localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhpDeg7].getWorkGroupSize(device);
            int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_BasicToCardinal2DXhpDeg7], width*height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhpDeg7],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhpDeg7].putArg(fullSizedGPUResidentHelperBuffer).putArg(entryImageBuffer).putArg(width).putArg(height);
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhpDeg7].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_BasicToCardinal2DYhpDeg7], width*height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhpDeg7],0,globalWorkSize,localWorkSize);
            // Start the reduction loop
            uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX].putArg(entryImageBuffer).putArg(fullSizedGPUResidentHelperBuffer).putArg(width).putArg(height).putArg(width/2);
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_reduceDual1DX], ((int)(width/2))*height);// Warning: integer division don't change
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX],0,globalWorkSize,localWorkSize);
            for(int j = 1;j < pyramidDepth; j++)
            {
                uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DY].putArg(fullSizedGPUResidentHelperBuffer).putArg(sourcePyramid[j].Image).putArg(width/2).putArg(height).putArg(height/2);
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DY].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_reduceDual1DY], ((int)(width/2))*((int)(height/2)));// Warning: integer division don't change
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DY],0,globalWorkSize,localWorkSize);
                if(j < pyramidDepth - 1)
                {
                    uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX].rewind();// reset argument index
                    uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX].putArg(sourcePyramid[j].Image).putArg(fullSizedGPUResidentHelperBuffer).putArg(width/2).putArg(height/2).putArg(((int)(width/2))/2 /*Warning integer division don't change*/ );
                    localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX].getWorkGroupSize(device);
                    globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_reduceDual1DX], ((int)(((int)(width/2))/2))*((int)(height/2)));// Warning: integer division don't change
                    queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX],0,globalWorkSize,localWorkSize);
                }
                width /= 2;
                height /= 2;
                // Restore the B-spline coefficients
                uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp].putArg(sourcePyramid[j].Image).putArg(width*height);
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DDeg7premulhp], width*height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXDeg7lp].putArg(sourcePyramid[j].Image).putArg(width).putArg(height);
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXDeg7lp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DXDeg7lp], height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXDeg7lp],0,globalWorkSize,localWorkSize);
                // pre-multiply again (the kernel is still setup correctly)
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DDeg7premulhp], width*height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYDeg7lp].putArg(sourcePyramid[j].Image).putArg(width).putArg(height);
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYDeg7lp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DYDeg7lp], width);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYDeg7lp],0,globalWorkSize,localWorkSize);
                // Now that we have the restored downsampled coefficients we still need to calculate the derivatives
                uniformBSplineTransformProgramKernels[KERNEL_antiSymmetricFirMirrorOffBounds1DX].putArg(sourcePyramid[j].Image).putArg(secondaryGPUResidentHelperBuffer).putArg(width).putArg(height);
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_antiSymmetricFirMirrorOffBounds1DX].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_antiSymmetricFirMirrorOffBounds1DX], width*height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_antiSymmetricFirMirrorOffBounds1DX],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_antiSymmetricFirMirrorOffBounds1DX].rewind();
                // Because all filters are linearly separable the Y coefficients may simply be restored on the X-diff and vice versa
                // TODO: possibly figure out if the half sized helper buffer couldn't be replaced by the entryImageBuffer
                uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhp].putArg(secondaryGPUResidentHelperBuffer).putArg(sourcePyramid[j].xGradient).putArg(width).putArg(height);
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_BasicToCardinal2DYhp], width*height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhp],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhp].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_antiSymmetricFirMirrorOffBounds1DY].putArg(sourcePyramid[j].Image).putArg(secondaryGPUResidentHelperBuffer).putArg(width).putArg(height);
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_antiSymmetricFirMirrorOffBounds1DY].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_antiSymmetricFirMirrorOffBounds1DY], width*height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_antiSymmetricFirMirrorOffBounds1DY],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_antiSymmetricFirMirrorOffBounds1DY].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhp].putArg(secondaryGPUResidentHelperBuffer).putArg(sourcePyramid[j].yGradient).putArg(width).putArg(height);
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_BasicToCardinal2DXhp], width*height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhp],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhp].rewind();
                // For the source image the downsampled image had to be restored from the B-spline coefficients 
                // and to avoid wasting memory the entry buffer will be used as intermediate buffer
                uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhp].putArg(sourcePyramid[j].Image).putArg(entryImageBuffer).putArg(width).putArg(height);
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_BasicToCardinal2DXhp], width*height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhp],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhp].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhp].putArg(entryImageBuffer).putArg(sourcePyramid[j].Image).putArg(width).putArg(height);
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_BasicToCardinal2DYhp], width*height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhp],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhp].rewind();
                // rewind the argument queues
                uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DY].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXDeg7lp].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYDeg7lp].rewind();
            }
            uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXlp].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYlp].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhpDeg7].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhpDeg7].rewind();
        }

        private void constructTargetImagePyramid()
        {
            int width = (int)sharedContext.img.dimension(0);
            int height = (int)sharedContext.img.dimension(1);
            if(!usesFloat)
            {
                // The double image is in the doubleEntryImageBuffer and the GPU can be used to calculate the coefficients
                int localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dTargetedCubicBSplinePrefilter2Dpremulhp].getWorkGroupSize(device);  // Local work size dimensions
                int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dTargetedCubicBSplinePrefilter2Dpremulhp], width*height);   // rounded up to the nearest multiple of the localWorkSize
                // pre-multiply the image
                uniformBSplineTransformProgramKernels[KERNEL_dTargetedCubicBSplinePrefilter2Dpremulhp].putArg(doubleEntryImageBuffer).putArg(targetDoubleSlice.Coefficient).putArg(width*height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dTargetedCubicBSplinePrefilter2Dpremulhp],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_dTargetedCubicBSplinePrefilter2Dpremulhp].rewind();
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DXhp].getWorkGroupSize(device);
                uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DXhp].putArg(targetDoubleSlice.Coefficient).putArg(width).putArg(height);
                // Conversion to B-spline coefficients along X axis (Group size must be >= height)
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dCubicBSplinePrefilter2DXhp], height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DXhp],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DXhp].rewind();
                // Now along the Y-axis
                // Has to be pre-multiplied by lambda again!!!
                uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2Dpremulhp].putArg(targetDoubleSlice.Coefficient).putArg(width*height);
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2Dpremulhp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dCubicBSplinePrefilter2Dpremulhp], width*height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2Dpremulhp],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2Dpremulhp].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DYhp].putArg(targetDoubleSlice.Coefficient).putArg(width).putArg(height);
                // Conversion to B-spline coefficients along Y axis (Group size must be >= width)
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DYhp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dCubicBSplinePrefilter2DYhp], width);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DYhp],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DYhp].rewind();
                // convert to float and copy
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_ConvertDoubleToFloat].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_ConvertDoubleToFloat], width*height);
                uniformBSplineTransformProgramKernels[KERNEL_ConvertDoubleToFloat].putArg(targetDoubleSlice.Coefficient).putArg(targetPyramid[0].Coefficient).putArg(width*height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_ConvertDoubleToFloat],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_ConvertDoubleToFloat].rewind();
            }
            else
            {
                // The double image is in the targetCoefficientDoubleSlice and the CPU needs to be used to calculate the coefficients
                // pre-multiply the image for cubic spline interpolation
                PlainJavaCPUAligner.premultiplyCubicBSpline(targetCoefficientDoubleSlice, width * height);
                // Conversion to B-spline coefficients along X axis
                PlainJavaCPUAligner.cubicBSplinePrefilter2DXhp(targetCoefficientDoubleSlice, width, height);
                // pre-multiply again
                PlainJavaCPUAligner.premultiplyCubicBSpline(targetCoefficientDoubleSlice, width * height);
                // Now along the Y-axis
                PlainJavaCPUAligner.cubicBSplinePrefilter2DYhp(targetCoefficientDoubleSlice, width, height);
                // convert to float and copy to the GPU
                ByteBuffer buffer = queue.putMapBuffer(targetPyramid[0].Coefficient, CLMemory.Map.WRITE, true);
                buffer.rewind();
                FloatBuffer fb = buffer.asFloatBuffer();
                for(int i = 0;i < targetCoefficientDoubleSlice.length;i++)
                {
                    fb.put((float)targetCoefficientDoubleSlice[i]);
                }
                queue.putUnmapMemory(targetPyramid[0].Coefficient, buffer);               
            }
            // TODO: use localWorkGroup size in a more sensible manner
            // Prepare the image for resampling by applying the FIR filter of degree 7 (out of place mod)
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhpDeg7].putArg(targetPyramid[0].Coefficient).putArg(fullSizedGPUResidentHelperBuffer).putArg(width).putArg(height);
            int localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhpDeg7].getWorkGroupSize(device);
            int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_BasicToCardinal2DXhpDeg7], width*height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhpDeg7],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhpDeg7].putArg(fullSizedGPUResidentHelperBuffer).putArg(entryImageBuffer).putArg(width).putArg(height);
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhpDeg7].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_BasicToCardinal2DYhpDeg7], width*height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhpDeg7],0,globalWorkSize,localWorkSize);
            // start the reduction loop
            uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX].putArg(entryImageBuffer).putArg(fullSizedGPUResidentHelperBuffer).putArg(width).putArg(height).putArg(width/2);
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_reduceDual1DX], ((int)(width/2))*height);// Warning: integer division don't change
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX],0,globalWorkSize,localWorkSize);
            for(int j = 1;j < pyramidDepth; j++)
            {
                uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DY].putArg(fullSizedGPUResidentHelperBuffer).putArg(targetPyramid[j].Coefficient).putArg(width/2).putArg(height).putArg(height/2);
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DY].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_reduceDual1DY], ((int)(width/2))*((int)(height/2)));// Warning: integer division don't change
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DY],0,globalWorkSize,localWorkSize);
                if(j < pyramidDepth - 1)
                {
                    uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX].rewind();// reset argument index
                    uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX].putArg(targetPyramid[j].Coefficient).putArg(fullSizedGPUResidentHelperBuffer).putArg(width/2).putArg(height/2).putArg(((int)(width/2))/2 /*Warning integer division don't change*/ );
                    localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX].getWorkGroupSize(device);
                    globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_reduceDual1DX], ((int)(((int)(width/2))/2))*((int)(height/2)));// Warning: integer division don't change
                    queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX],0,globalWorkSize,localWorkSize);
                }
                width /= 2;
                height /= 2;
                // restore the B-spline coefficients
                uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp].putArg(targetPyramid[j].Coefficient).putArg(width*height);
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DDeg7premulhp], width*height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXDeg7lp].putArg(targetPyramid[j].Coefficient).putArg(width).putArg(height);
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXDeg7lp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DXDeg7lp], height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXDeg7lp],0,globalWorkSize,localWorkSize);
                // pre-multiply again (the kernel is still set up correctly)
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DDeg7premulhp], width*height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYDeg7lp].putArg(targetPyramid[j].Coefficient).putArg(width).putArg(height);
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYDeg7lp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DYDeg7lp], width);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYDeg7lp],0,globalWorkSize,localWorkSize);
                // Now we have the restored downsampled coefficients and they are already in the pyramid storage so no need to copy them just rewind the argument queues
                uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DY].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXDeg7lp].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYDeg7lp].rewind();
            }
            uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXlp].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYlp].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhpDeg7].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhpDeg7].rewind();
        }
        private void putSourceImageIntoPipelineEntry()
        {
            if(!usesFloat)
            {
                // Due to some bug writing these huge buffers to memory always crashes OpenCL, mapping the memory on the other hand works fine
                if ((sharedContext.img.firstElement() instanceof ByteType)||(sharedContext.img.firstElement() instanceof UnsignedByteType)) {
                    ByteBuffer buffer = queue.putMapBuffer(conversionEntryBuffer, CLMemory.Map.WRITE, true);
                    buffer.rewind();
                    buffer.put((byte[]) scat.sourceArray);
                    queue.putUnmapMemory(conversionEntryBuffer, buffer);
                } else if ((sharedContext.img.firstElement() instanceof ShortType)||(sharedContext.img.firstElement() instanceof UnsignedShortType)) {
                    ByteBuffer buffer = queue.putMapBuffer(conversionEntryBuffer, CLMemory.Map.WRITE, true);
                    buffer.rewind();
                    buffer.asShortBuffer().put((short[]) scat.sourceArray);
                    queue.putUnmapMemory(conversionEntryBuffer, buffer);
                } else if ((sharedContext.img.firstElement() instanceof IntType)||(sharedContext.img.firstElement() instanceof UnsignedIntType)) {
                    ByteBuffer buffer = queue.putMapBuffer(conversionEntryBuffer, CLMemory.Map.WRITE, true);
                    buffer.rewind();
                    buffer.asIntBuffer().put((int[]) scat.sourceArray);
                    queue.putUnmapMemory(conversionEntryBuffer, buffer);
                } else if ((sharedContext.img.firstElement() instanceof FloatType)) {
                    // Conversion buffers are only needed if the representation is double later on
                    ByteBuffer buffer = queue.putMapBuffer(conversionEntryBuffer, CLMemory.Map.WRITE, true);
                    buffer.rewind();
                    buffer.asFloatBuffer().put((float[]) scat.sourceArray);
                    queue.putUnmapMemory(conversionEntryBuffer, buffer);
                }
                int localWorkSize = (int)conversionProgramKernel.getWorkGroupSize(device);  // Local work size dimensions
                int globalWorkSize = StaticUtility.roundUp(localWorkSize, conversionProgramKernel.getPreferredWorkGroupSizeMultiple(device), (int)sharedContext.img.dimension(0)*(int)sharedContext.img.dimension(1));   // rounded up to the nearest multiple of the localWorkSize
                queue.put1DRangeKernel(conversionProgramKernel,0,globalWorkSize,localWorkSize);
                queue.putCopyBuffer(doubleEntryImageBuffer, sourceDoubleSlice.Image);
                // convert to float and copy
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_ConvertDoubleToFloat].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_ConvertDoubleToFloat], (int)sharedContext.img.dimension(0)*(int)sharedContext.img.dimension(1));
                uniformBSplineTransformProgramKernels[KERNEL_ConvertDoubleToFloat].putArg(doubleEntryImageBuffer).putArg(sourcePyramid[0].Image).putArg((int)sharedContext.img.dimension(0)*(int)sharedContext.img.dimension(1));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_ConvertDoubleToFloat],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_ConvertDoubleToFloat].rewind();
            }
            else
            {
                converter.convertTo(scat.sourceArray, CPUentryImageBuffer); 
                System.arraycopy(CPUentryImageBuffer, 0, sourceImageDoubleSlice, 0, (int)sharedContext.img.dimension(0)*(int)sharedContext.img.dimension(1)); //copy
                // convert to float and copy
                ByteBuffer buffer = queue.putMapBuffer(sourcePyramid[0].Image, CLMemory.Map.WRITE, true);
                buffer.rewind();
                FloatBuffer fb = buffer.asFloatBuffer();
                for(int i = 0;i < CPUentryImageBuffer.length;i++)
                {
                    fb.put((float)CPUentryImageBuffer[i]);
                }
                queue.putUnmapMemory(sourcePyramid[0].Image, buffer);
            }
        }
        private void putTargetImageIntoPipelineEntry()
        {
            if(!usesFloat)
            {
                // Due to some bug writing these huge buffers to memory always crashes OpenCL, mapping the memory on the other hand works fine
                if ((sharedContext.img.firstElement() instanceof ByteType)||(sharedContext.img.firstElement() instanceof UnsignedByteType)) {
                    ByteBuffer buffer = queue.putMapBuffer(conversionEntryBuffer, CLMemory.Map.WRITE, true);
                    buffer.rewind();
                    buffer.put((byte[]) scat.targetArray);
                    queue.putUnmapMemory(conversionEntryBuffer, buffer);
                } else if ((sharedContext.img.firstElement() instanceof ShortType)||(sharedContext.img.firstElement() instanceof UnsignedShortType)) {
                    ByteBuffer buffer = queue.putMapBuffer(conversionEntryBuffer, CLMemory.Map.WRITE, true);
                    buffer.rewind();
                    buffer.asShortBuffer().put((short[]) scat.targetArray);
                    queue.putUnmapMemory(conversionEntryBuffer, buffer);
                } else if ((sharedContext.img.firstElement() instanceof IntType)||(sharedContext.img.firstElement() instanceof UnsignedIntType)) {
                    ByteBuffer buffer = queue.putMapBuffer(conversionEntryBuffer, CLMemory.Map.WRITE, true);
                    buffer.rewind();
                    buffer.asIntBuffer().put((int[]) scat.targetArray);
                    queue.putUnmapMemory(conversionEntryBuffer, buffer);
                } else if ((sharedContext.img.firstElement() instanceof FloatType)) {
                	// Conversion buffers are only needed if the representation is double later on
                    ByteBuffer buffer = queue.putMapBuffer(conversionEntryBuffer, CLMemory.Map.WRITE, true);
                    buffer.rewind();
                    buffer.asFloatBuffer().put((float[]) scat.targetArray);
                    queue.putUnmapMemory(conversionEntryBuffer, buffer);
                }
                int localWorkSize = (int)conversionProgramKernel.getWorkGroupSize(device);  // Local work size dimensions
                int globalWorkSize = StaticUtility.roundUp(localWorkSize, conversionProgramKernel.getPreferredWorkGroupSizeMultiple(device),(int) sharedContext.img.dimension(0)*(int)sharedContext.img.dimension(1));   // rounded up to the nearest multiple of the localWorkSize
                queue.put1DRangeKernel(conversionProgramKernel,0,globalWorkSize,localWorkSize);
            }
            else
            {
                converter.convertTo(scat.targetArray, targetCoefficientDoubleSlice); //can be modified in place later on
            }
        }
        public void start() {
            // Unfortunately java does not allow restarting (and therefore reusing) a thread so here it is created again
            // see http://docs.oracle.com/javase/7/docs/api/java/lang/Thread.html "It is never legal to start a thread more than once.
        	// In particular, a thread may not be restarted once it has completed execution."
            t = new Thread(this);//always use a new thread
            t.start();
        }
        Thread getThread() {
            return t;
        }
    }

    public HybridPrecisionNGStackReg(boolean permitsFloatGPU, final AbstractSharedContext sharedContext) throws Exception
    {
        super(sharedContext);
        if(this.sharedContext.forceDoublePrecisionRepr)
        {
            this.permitsFloatGPU = false;
        }
        else
        {
            this.permitsFloatGPU = permitsFloatGPU;
        }
        calculatePyramidDepth();
        enumerateOCLDevicesAndInitialize();
        sharedContext.addParties(workers.length);
    }
    private void enumerateOCLDevicesAndInitialize() throws Exception
    {
        /*
        Created command queues in in-order execution by not setting out-of-order
        execution. According to the docs this allows to call multiple kernels
        after one another without cross synchronization
        */
        if(this.permitsFloatGPU)
        {
            // In case double support is not required
            CLPlatform[] platforms = CLPlatform.listCLPlatforms();
            List<CLDevice> listOfdevices = new ArrayList<>();
            for(CLPlatform p: platforms)
            {
                CLDevice[] devs = p.listCLDevices(CLDevice.Type.GPU);// only take GPU devices
                listOfdevices.addAll(Arrays.asList(devs));
            }
            if(listOfdevices.size() <= 0)
            {
                throw new Exception("Could not find a GPU device");
            }
            contexts = new CLContext[listOfdevices.size()];
            if(contexts == null)
            {
                throw new Exception("Could not perform allocate CLContext's");
            }
            int localcounter = 0;
            for(CLDevice d : listOfdevices)
            {
                contexts[localcounter] = CLContext.create(d);// Create a context on the specified device
                localcounter++;
            }
            devices = listOfdevices.toArray(new CLDevice[0]);
        }
        else
        {
            // only choose devices which support double
            CLPlatform[] platforms = CLPlatform.listCLPlatforms();
            List<CLDevice> listOfdevices = new ArrayList<>();
            for(CLPlatform p: platforms)
            {
                CLDevice[] devs = p.listCLDevices(CLDevice.Type.GPU);// only take GPU devices
                for(CLDevice d: devs)
                {
                    if(d.isDoubleFPAvailable())
                    {
                        listOfdevices.add(d);
                    }
                }
            }
            if(listOfdevices.size() <= 0)
            {
                throw new Exception("Could not find a GPU device that supports double");
            }
            contexts = new CLContext[listOfdevices.size()];
            if(contexts == null)
            {
                throw new Exception("Could not perform allocate CLContext's");
            }
            int localcounter = 0;
            for(CLDevice d : listOfdevices)
            {
                contexts[localcounter] = CLContext.create(d);// Create a context on the specified device
                localcounter++;
            }
            devices = listOfdevices.toArray(new CLDevice[0]);
        }
        int nrOfWorkers = devices.length;
        workers = new HybridPrecisionNGStackRegWorker[nrOfWorkers];
        for(int i = 0; i < nrOfWorkers; i++)
        {
        	workers[i] = new HybridPrecisionNGStackRegWorker(contexts[i],devices[i]);
        }
        this.sharedContext.nrOfGPUDevices = workers.length;
    }
    private void calculatePyramidDepth()
    {
        long s = sharedContext.img.dimension(0) < sharedContext.img.dimension(1) ? sharedContext.img.dimension(0) : sharedContext.img.dimension(1);
        while (s >= NGStackReg.MIN_SIZE) {
            s /= 2;
            pyramidDepth++;
        }
    }

    @Override
    public void release()
    {
        if(contexts != null)
        {
        	for(CLContext c : contexts)
        	{
        		if(c != null)
                {
                    c.release();
                }
        	}
        }
    }
    @Override
    public void register() throws InterruptedException, Exception
    {
        for(HybridPrecisionNGStackRegWorker worker: workers)
        {
        	worker.start();
        }
    }
    
    @Override
    public void waitForFinish()  throws InterruptedException {
        if(workers != null)
        {
            for(HybridPrecisionNGStackRegWorker worker: workers)
            {
            	if(worker != null)
                {
                    worker.getThread().join();
                }
            }
        }
    }
}
