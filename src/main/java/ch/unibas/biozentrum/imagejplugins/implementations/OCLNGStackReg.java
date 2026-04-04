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
import ch.unibas.biozentrum.imagejplugins.abstracts.RegistrationAndTransformation;
import ch.unibas.biozentrum.imagejplugins.util.AffineTransformation;
import ch.unibas.biozentrum.imagejplugins.util.RigidBodyTransformation;
import ch.unibas.biozentrum.imagejplugins.util.ScaledRotationTransformation;
import ch.unibas.biozentrum.imagejplugins.util.TranslationTransformation;
import ch.unibas.biozentrum.imagejplugins.util.StaticUtility;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory;
import com.jogamp.opencl.CLPlatform;
import com.jogamp.opencl.CLProgram;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BrokenBarrierException;
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

/**
 * @author Peter D. Ringel
 * @version 1.0.0
 *
 */
class OCLSourcePyramidSlice
{
    long width;
    long height;
    @SuppressWarnings("rawtypes")
    CLBuffer Image;
    @SuppressWarnings("rawtypes")
    CLBuffer xGradient;
    @SuppressWarnings("rawtypes")
    CLBuffer yGradient;
}
class OCLTargetPyramidSlice
{
    long width;
    long height;
    @SuppressWarnings("rawtypes")
    CLBuffer Coefficient;
}
public class OCLNGStackReg extends RegistrationAndTransformation
{
    // Just constants for accessing the compiled kernels in the program (compiled separately for each device)
    private static final int NR_OF_OPENCL_KERNELS = 23;
    private static final int KERNEL_CubicBSplinePrefilter2Dpremulhp = 0;
    private static final int KERNEL_CubicBSplinePrefilter2DXhp = 1;
    private static final int KERNEL_CubicBSplinePrefilter2DYhp = 2;
    private static final int KERNEL_BasicToCardinal2DXhp = 3;
    private static final int KERNEL_BasicToCardinal2DYhp = 4;
    private static final int KERNEL_CubicBSplinePrefilter2DDeg7premulhp = 5;
    private static final int KERNEL_CubicBSplinePrefilter2DXDeg7hp = 6;
    private static final int KERNEL_CubicBSplinePrefilter2DYDeg7hp = 7;
    private static final int KERNEL_BasicToCardinal2DXhpDeg7 = 8;
    private static final int KERNEL_BasicToCardinal2DYhpDeg7 = 9;
    private static final int KERNEL_reduceDual1DX = 10;
    private static final int KERNEL_reduceDual1DY = 11;
    private static final int KERNEL_antiSymmetricFirMirrorOffBounds1DX = 12;
    private static final int KERNEL_antiSymmetricFirMirrorOffBounds1DY = 13;
    
    private static final int KERNEL_sumInLocalMemory = 17;
    private static final int KERNEL_parallelGroupedSumReduction = 18;
    private static final int KERNEL_TargetedCubicBSplinePrefilter2Dpremulhp = 14;
    
    private static final int KERNEL_rigidBodyError = 15;
    private static final int KERNEL_rigidBodyErrorWithGradAndHess = 16;
    private static final int KERNEL_sumInLocalMemoryCombined = 19;
    private static final int KERNEL_rigidBodyErrorWithGradAndHessBrent = 20;
    private static final int KERNEL_transformImageWithBsplineInterpolation = 21;
    private static final int KERNEL_resizeTransformImageWithBsplineInterpolation = 22;
    
    private static final int KERNEL_translationError = 15; // KERNEL_rigidBodyError
    private static final int KERNEL_translationErrorWithGradAndHess = 16; // KERNEL_rigidBodyErrorWithGradAndHess
    private static final int KERNEL_translationErrorWithGradAndHessBrent = 20; // KERNEL_rigidBodyErrorWithGradAndHessBrent
    private static final int KERNEL_translationTransformImageWithBsplineInterpolation = 21; // KERNEL_transformImageWithBsplineInterpolation
    private static final int KERNEL_translationSumInLocalMemoryCombined = 19; // KERNEL_sumInLocalMemoryCombined
    
    private static final int KERNEL_scaledRotationError = 15; // KERNEL_rigidBodyError
    private static final int KERNEL_scaledRotationErrorWithGradAndHess = 16; // KERNEL_rigidBodyErrorWithGradAndHess
    private static final int KERNEL_scaledRotationErrorWithGradAndHessBrent = 20; // KERNEL_rigidBodyErrorWithGradAndHessBrent
    private static final int KERNEL_scaledRotationTransformImageWithBsplineInterpolation = 21; // KERNEL_transformImageWithBsplineInterpolation
    private static final int KERNEL_scaledRotationSumInLocalMemoryCombined = 19; // KERNEL_sumInLocalMemoryCombined
    
    private static final int KERNEL_affineError = 15; // KERNEL_rigidBodyError
    private static final int KERNEL_affineErrorWithGradAndHess = 16; // KERNEL_rigidBodyErrorWithGradAndHess
    private static final int KERNEL_affineErrorWithGradAndHessBrent = 20; // KERNEL_rigidBodyErrorWithGradAndHessBrent
    private static final int KERNEL_affineTransformImageWithBsplineInterpolation = 21; // KERNEL_transformImageWithBsplineInterpolation
    private static final int KERNEL_affineSumInLocalMemoryCombined = 19; // KERNEL_sumInLocalMemoryCombined
    
    private static final int maximumSumReductionBlockNr = 64;// a maximum of 64 kernel blocks with a variable width will be started
    private static final int blocksizeMultiplier = 4;// optimal multiple * blocksizeMultiplier = blockSizes if this is less than the maximum number of elements that can be accommodated
    // Because the acquired devices can be very heterogeneous the block sizes have to be kept separate (they should be a multiple of the optimal multiple size)
    private final boolean useFloatGPUOnly;
    private boolean usesFloatGPU = false;
    private CLContext[] contexts;
    private CLDevice[] devices = null;
    private int pyramidDepth = 1;
    
    private static final CLMemory.Mem[] GPURESIDENTRW = {CLMemory.Mem.READ_WRITE};
    private OCLNGStackRegWorker[] workers;
    /*
    Use an internal class because this has access to the parents attributes
    without having to explicitly pass them along.
    */
    private class OCLNGStackRegWorker implements Runnable
    {
        /*
        This is the helper class encapsulating the code executed per thread.
        It implements the whole registration and transformation code using a
        work stealing approach because, if there are different GPUs available
        on the system, they may not be equivalent in their processing speed,
        which would lead to waiting for the slowest one when splitting the work
        equally.
        */
        private Thread t = null;
        private double offsetx = 0.0;
        private double offsety = 0.0;
        private double angle = 0.0;
        private double scale = 1.0;
        private double a11 = 1.0;
        private double a12 = 0.0;
        private double a21 = 0.0;
        private double a22 = 1.0;
        private double[][] hessian;
        private double[][] pseudoHessian;
        private double[] gradient;
        private int iterationPower;
        
        private int blockSizesFPT;
        private int blockSizesParallel;
        private long optimalMultiples[];
        private int maximumElementsForLocalFPTsum;
        private int maximumElementsForLocalFPTcombinedSum;
        
        private final CLContext context;
        private final CLDevice device;
        private CLCommandQueue queue = null;
        private CLCommandQueue asyncQueue = null;
        
        private OCLSourcePyramidSlice[] sourcePyramid;
        private OCLTargetPyramidSlice[] targetPyramid;
        @SuppressWarnings("rawtypes")
        private CLBuffer conversionEntryBuffer;
        @SuppressWarnings("rawtypes")
        private CLBuffer entryImageBuffer;
        @SuppressWarnings("rawtypes")
        private CLBuffer fullSizedGPUResidentHelperBuffer;
        @SuppressWarnings("rawtypes")
        private CLBuffer secondaryGPUResidentHelperBuffer;
        
        @SuppressWarnings("rawtypes")
        private CLBuffer[] parallelSumReductionBuffers;
        // These buffers are not necessary but the decision was to squeeze out more performance while using more memory
        @SuppressWarnings("rawtypes")
        private CLBuffer gradient0;
        @SuppressWarnings("rawtypes")
        private CLBuffer gradient1;
        @SuppressWarnings("rawtypes")
        private CLBuffer gradient2;
        @SuppressWarnings("rawtypes")
        private CLBuffer gradient3;
        @SuppressWarnings("rawtypes")
        private CLBuffer gradient4;
        @SuppressWarnings("rawtypes")
        private CLBuffer gradient5;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian00;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian01;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian02;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian03;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian04;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian05;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian11;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian12;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian13;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian14;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian15;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian22;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian23;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian24;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian25;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian33;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian34;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian35;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian44;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian45;
        @SuppressWarnings("rawtypes")
        private CLBuffer hessian55;
        @SuppressWarnings("rawtypes")
        private CLBuffer maskBuffer;
        
        private CLProgram conversionProgram;
        private CLKernel conversionProgramKernel;
        private CLKernel deConversionProgramKernel;
        
        private CLProgram uniformBSplineTransformProgram;
        private CLKernel uniformBSplineTransformProgramKernels[];

        private final SharedContextAlignmentTarget scat = new SharedContextAlignmentTarget();
        OCLNGStackRegWorker(final CLContext context, final CLDevice device) throws Exception
        {
            switch(sharedContext.transformationType) {
            case TRANSLATION:
                hessian = new double[][] { {0.0,0.0},{0.0,0.0} };
                pseudoHessian = new double[][] { {0.0,0.0},{0.0,0.0} };
                gradient = new double[] {0.0,0.0};
                break;
            case RIGIDBODY:
                hessian = new double[][] { {0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0} };
                pseudoHessian = new double[][] { {0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0} };
                gradient = new double[] {0.0,0.0,0.0};
                break;
            case SCALEDROTATION:
                gradient = new double[]{0.0, 0.0, 0.0, 0.0};
                hessian = new double[][]{{0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0}};
                pseudoHessian = new double[][]{{0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0}};
                break;
            case AFFINE:
                gradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                hessian = new double[][]{{0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}};
                pseudoHessian = new double[][]{{0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}};
                break;
            default:
                hessian = new double[][] { {0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0} };
                pseudoHessian = new double[][] { {0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0} };
                gradient = new double[] {0.0,0.0,0.0};
            }
            this.context = context;
            this.device = device;
            enumerateOCLDevicesAndInitialize();
            allocateMemory();
            CompileAndSetupOpenCLKernerls();
        }
        
        private void cleanup()
        {
            if(deConversionProgramKernel != null)
            {
            	deConversionProgramKernel.release();
				deConversionProgramKernel = null;
            }
            if(conversionProgramKernel != null)
            {
            	conversionProgramKernel.release();
                conversionProgramKernel = null;
            }
            if(conversionProgram != null)
			{
				conversionProgram.release();
				conversionProgram = null;
			}
            for(int i = 0; i < uniformBSplineTransformProgramKernels.length; i++)
			{
				if(uniformBSplineTransformProgramKernels[i] != null)
				{
					uniformBSplineTransformProgramKernels[i].release();
					uniformBSplineTransformProgramKernels[i] = null;
				}
			}
            if(uniformBSplineTransformProgram != null)
            {
            	uniformBSplineTransformProgram.release();
            	uniformBSplineTransformProgram = null;
            }
        	
            switch(sharedContext.transformationType) {
            case TRANSLATION:
            	if(maskBuffer != null)
            	{
            		maskBuffer.release();
					maskBuffer = null;
            	}
                if(gradient0 != null)
				{
					gradient0.release();
					gradient0 = null;
				}
                if(gradient1 != null)
                {
                	gradient1.release();
                	gradient1 = null;
                }
                if(hessian00 != null)
				{
					hessian00.release();
					hessian00 = null;
				}
                if(hessian01 != null)
				{
					hessian01.release();
					hessian01 = null;
				}
                if(hessian11 != null)
                {
                	hessian11.release();
					hessian11 = null;
                }
                break;
            case RIGIDBODY:
            	if(maskBuffer != null)
            	{
            		maskBuffer.release();
            		maskBuffer = null;
            	}
            	if(gradient0 != null)
            	{
            		gradient0.release();
					gradient0 = null;
            	}
            	if(gradient1 != null)
            	{
            		gradient1.release();
            		gradient1 = null;
            	}
            	if(gradient2 != null)
            	{
            		gradient2.release();
					gradient2 = null;
            	}
            	if(hessian00 != null)
            	{
            		hessian00.release();
            		hessian00 = null;
            	}
            	if(hessian01 != null)
            	{
            		hessian01.release();
					hessian01 = null;
            	}
            	if(hessian02 != null)
            	{
            		hessian02.release();
            		hessian02 = null;
            	}
            	if(hessian11 != null)
            	{
            		hessian11.release();
            		hessian11 = null;
            	}
            	if(hessian12 != null)
            	{
            		hessian12.release();
					hessian12 = null;
            	}
            	if(hessian22 != null)
            	{
            		hessian22.release();
					hessian22 = null;
            	}
                break;
            case SCALEDROTATION:
            	if(maskBuffer != null)
            	{
            		maskBuffer.release();
            		maskBuffer = null;
            	}
            	if(gradient0 != null)
            	{
            		gradient0.release();
            		gradient0 = null;
            	}
            	if(gradient1 != null)
            	{
            		gradient1.release();
					gradient1 = null;
            	}
            	if(gradient2 != null)
            	{
            		gradient2.release();
            		gradient2 = null;
            	}
            	if(gradient3 != null)
            	{
            		gradient3.release();
            		gradient3 = null;
            	}
            	if(hessian00 != null)
            	{
            		hessian00.release();
					hessian00 = null;
            	}
            	if(hessian01 != null)
            	{
            		hessian01.release();
            		hessian01 = null;
            	}
            	if(hessian02 != null)
            	{
            		hessian02.release();
            		hessian02 = null;
            	}
            	if(hessian03 != null)
            	{
            		hessian03.release();
					hessian03 = null;
            	}
            	if(hessian11 != null)
            	{
            		hessian11.release();
					hessian11 = null;
            	}
            	if(hessian12 != null)
            	{
            		hessian12.release();
            		hessian12 = null;
            	}
            	if(hessian13 != null)
            	{
            		hessian13.release();
            		hessian13 = null;
            	}
            	if(hessian22 != null)
            	{
            		hessian22.release();
            		hessian22 = null;
            	}
            	if(hessian23 != null)
            	{
            		hessian23.release();
					hessian23 = null;
            	}
            	if(hessian33 != null)
            	{
            		hessian33.release();
            		hessian33 = null;
            	}
                break;
            case AFFINE:
            	if(maskBuffer != null)
            	{
            		maskBuffer.release();
					maskBuffer = null;
            	}
            	if(gradient0 != null)
            	{
            		gradient0.release();
            		gradient0 = null;
            	}
            	if(gradient1 != null)
            	{
            		gradient1.release();
            		gradient1 = null;
            	}
            	if(gradient2 != null)
            	{
            		gradient2.release();
					gradient2 = null;
            	}
            	if(gradient3 != null)
            	{
            		gradient3.release();
					gradient3 = null;
            	}
            	if(gradient4 != null)
            	{
            		gradient4.release();
            		gradient4 = null;
            	}
            	if(gradient5 != null)
            	{
            		gradient5.release();
					gradient5 = null;
            	}
            	if(hessian00 != null)
            	{
            		hessian00.release();
            		hessian00 = null;
            	}
            	if(hessian01 != null)
            	{
            		hessian01.release();
            		hessian01 = null;
            	}
            	if(hessian02 != null)
            	{
            		hessian02.release();
					hessian02 = null;
            	}
            	if(hessian03 != null)
            	{
            		hessian03.release();
            		hessian03 = null;
            	}
            	if(hessian04 != null)
            	{
            		hessian04.release();
					hessian04 = null;
            	}
            	if(hessian05 != null)
            	{
            		hessian05.release();
					hessian05 = null;
            	}
            	if(hessian11 != null)
            	{
            		hessian11.release();
            		hessian11 = null;
            	}
            	if(hessian12 != null)
            	{
            		hessian12.release();
            		hessian12 = null;
            	}
            	if(hessian13 != null)
            	{
            		hessian13.release();
            		hessian13 = null;
            	}
            	if(hessian14 != null)
            	{
            		hessian14.release();
            		hessian14 = null;
            	}
            	if(hessian15 != null)
            	{
            		hessian15.release();
            		hessian15 = null;
            	}
            	if(hessian22 != null)
            	{
            		hessian22.release();
					hessian22 = null;
            	}
            	if(hessian23 != null)
            	{
            		hessian23.release();
					hessian23 = null;
            	}
            	if(hessian24 != null)
            	{
            		hessian24.release();
            		hessian24 = null;
            	}
            	if(hessian25 != null)
            	{
            		hessian25.release();
            		hessian25 = null;
            	}
            	if(hessian33 != null)
            	{
            		hessian33.release();
					hessian33 = null;
            	}
            	if(hessian34 != null)
            	{
            		hessian34.release();
					hessian34 = null;
            	}
            	if(hessian35 != null)
            	{
            		hessian35.release();
            		hessian35 = null;
            	}
            	if(hessian44 != null)
            	{
            		hessian44.release();
            		hessian44 = null;
            	}
            	if(hessian45 != null)
            	{
            		hessian45.release();
            		hessian45 = null;
            	}
            	if(hessian55 != null)
            	{
            		hessian55.release();
					hessian55 = null;
            	}
                break;
            }
        	for(int l = 0;l < 2;l++)
            {
        		if(parallelSumReductionBuffers[l] != null)
        		{
        			parallelSumReductionBuffers[l].release();
        			parallelSumReductionBuffers[l] = null;
        		}
            }
            for(int j = 0;j < pyramidDepth; j++)
            {
            	if(sourcePyramid[j].Image != null)
            	{
            		sourcePyramid[j].Image.release();
            		sourcePyramid[j].Image = null;
            	}
            	if(sourcePyramid[j].xGradient != null)
            	{
            		sourcePyramid[j].xGradient.release();
					sourcePyramid[j].xGradient = null;
            	}
            	if(sourcePyramid[j].yGradient != null)
            	{
            		sourcePyramid[j].yGradient.release();
            		sourcePyramid[j].yGradient = null;
            	}
            	if(targetPyramid[j].Coefficient != null)
            	{
            		targetPyramid[j].Coefficient.release();
					targetPyramid[j].Coefficient = null;
            	}
            }
            
            if(fullSizedGPUResidentHelperBuffer != null)
		    {
            	fullSizedGPUResidentHelperBuffer.release();
            	fullSizedGPUResidentHelperBuffer = null;
		    }
            if(secondaryGPUResidentHelperBuffer != null)
            {
            	secondaryGPUResidentHelperBuffer.release();
				secondaryGPUResidentHelperBuffer = null;
            }
            if(entryImageBuffer != null)
			{
				entryImageBuffer.release();
				entryImageBuffer = null;
			}
            if(conversionEntryBuffer != null)
            {
            	conversionEntryBuffer.release();
				conversionEntryBuffer = null;
            }
            
            if(queue != null)
			{
				queue.release();
				queue = null;
			}
            if(asyncQueue != null)
			{
				asyncQueue.release();
				asyncQueue = null;
			}
        }
        
        private void enumerateOCLDevicesAndInitialize() throws Exception
        {
            /*
            Created command queues in in-order execution by not setting out-of-order
            execution. According to the docs this allows one to call multiple kernels
            after one another without cross synchronization
            */
            queue = device.createCommandQueue();
            if(device.getQueueProperties().contains(CLCommandQueue.Mode.OUT_OF_ORDER_MODE))
            {
                // Make it async if possible
                asyncQueue = device.createCommandQueue(CLCommandQueue.Mode.OUT_OF_ORDER_MODE);
            }
            else
            {
                asyncQueue = device.createCommandQueue();
            }
        }
        
        private void resizeAllocatedBuffers()
        {
            // First release all the memory that isn't needed anymore
            switch(sharedContext.transformationType) {
            case TRANSLATION:
                maskBuffer.release();
                maskBuffer = null;
                gradient0.release();
                gradient0 = null;
                gradient1.release();
                gradient1 = null;
                hessian00.release();
                hessian00 = null;
                hessian01.release();
                hessian01 = null;
                hessian11.release();
                hessian11 = null;
                break;
            case RIGIDBODY:
                maskBuffer.release();
                maskBuffer = null;
                gradient0.release();
                gradient0 = null;
                gradient1.release();
                gradient1 = null;
                gradient2.release();
                gradient2 = null;
                hessian00.release();
                hessian00 = null;
                hessian01.release();
                hessian01 = null;
                hessian02.release();
                hessian02 = null;
                hessian11.release();
                hessian11 = null;
                hessian12.release();
                hessian12 = null;
                hessian22.release();
                hessian22 = null;
                break;
            case SCALEDROTATION:
                maskBuffer.release();
                maskBuffer = null;
                gradient0.release();
                gradient0 = null;
                gradient1.release();
                gradient1 = null;
                gradient2.release();
                gradient2 = null;
                gradient3.release();
                gradient3 = null;
                hessian00.release();
                hessian00 = null;
                hessian01.release();
                hessian01 = null;
                hessian02.release();
                hessian02 = null;
                hessian03.release();
                hessian03 = null;
                hessian11.release();
                hessian11 = null;
                hessian12.release();
                hessian12 = null;
                hessian13.release();
                hessian13 = null;
                hessian22.release();
                hessian22 = null;
                hessian23.release();
                hessian23 = null;
                hessian33.release();
                hessian33 = null;
                break;
            case AFFINE:
                maskBuffer.release();
                maskBuffer = null;
                gradient0.release();
                gradient0 = null;
                gradient1.release();
                gradient1 = null;
                gradient2.release();
                gradient2 = null;
                gradient3.release();
                gradient3 = null;
                gradient4.release();
                gradient4 = null;
                gradient5.release();
                gradient5 = null;
                hessian00.release();
                hessian00 = null;
                hessian01.release();
                hessian01 = null;
                hessian02.release();
                hessian02 = null;
                hessian03.release();
                hessian03 = null;
                hessian04.release();
                hessian04 = null;
                hessian05.release();
                hessian05 = null;
                hessian11.release();
                hessian11 = null;
                hessian12.release();
                hessian12 = null;
                hessian13.release();
                hessian13 = null;
                hessian14.release();
                hessian14 = null;
                hessian15.release();
                hessian15 = null;
                hessian22.release();
                hessian22 = null;
                hessian23.release();
                hessian23 = null;
                hessian24.release();
                hessian24 = null;
                hessian25.release();
                hessian25 = null;
                hessian33.release();
                hessian33 = null;
                hessian34.release();
                hessian34 = null;
                hessian35.release();
                hessian35 = null;
                hessian44.release();
                hessian44 = null;
                hessian45.release();
                hessian45 = null;
                hessian55.release();
                hessian55 = null;
                break;
            }
            for(int l = 0;l < 2;l++)
            {
                parallelSumReductionBuffers[l].release();
                parallelSumReductionBuffers[l] = null;
            }
            for(int j = 0;j < pyramidDepth; j++)
            {
                sourcePyramid[j].Image.release();
                sourcePyramid[j].Image = null;
                sourcePyramid[j].xGradient.release();
                sourcePyramid[j].xGradient = null;
                sourcePyramid[j].yGradient.release();
                sourcePyramid[j].yGradient = null;
                targetPyramid[j].Coefficient.release();
                targetPyramid[j].Coefficient = null;
            }
            
            /*
             * The target buffer has to be resized
             * entryImageBuffer <- source (can stay the same)
             * fullSizedGPUResidentHelperBuffer <- target (must be resized)
             * then the deconversion
             * fullSizedGPUResidentHelperBuffer <- source
             * conversionEntryBuffer <- target (must be resized)
             */
            fullSizedGPUResidentHelperBuffer.release();
            fullSizedGPUResidentHelperBuffer = null;
            secondaryGPUResidentHelperBuffer.release();
            secondaryGPUResidentHelperBuffer = null;
            if(usesFloatGPU)
            {
                fullSizedGPUResidentHelperBuffer = context.createFloatBuffer((int) (sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)), GPURESIDENTRW);
            }
            else
            {
                fullSizedGPUResidentHelperBuffer = context.createDoubleBuffer((int) (sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)), GPURESIDENTRW);
            }
            
            // Check if the new image is larger then reallocate, otherwise keep the old buffer
            if((sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)) > (sharedContext.img.dimension(0)*sharedContext.img.dimension(1)))
            {
                conversionEntryBuffer.release();
                conversionEntryBuffer = null;
                
                // Really ugly code but I couldn't figure out how to do this more elegantly
                if((sharedContext.resizedTargetImage.firstElement() instanceof ByteType)||(sharedContext.resizedTargetImage.firstElement() instanceof UnsignedByteType))
                {
                    conversionEntryBuffer = context.createByteBuffer((int) (sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)), GPURESIDENTRW);
                }
                else if((sharedContext.resizedTargetImage.firstElement() instanceof ShortType)||(sharedContext.resizedTargetImage.firstElement() instanceof UnsignedShortType))
                {
                    conversionEntryBuffer = context.createShortBuffer((int) (sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)), GPURESIDENTRW);
                }
                else if((sharedContext.resizedTargetImage.firstElement() instanceof IntType)||(sharedContext.resizedTargetImage.firstElement() instanceof UnsignedIntType))
                {
                    conversionEntryBuffer = context.createIntBuffer((int) (sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)), GPURESIDENTRW);
                }
                else if((sharedContext.resizedTargetImage.firstElement() instanceof LongType)||(sharedContext.resizedTargetImage.firstElement() instanceof UnsignedLongType))
                {
                    conversionEntryBuffer = context.createLongBuffer((int) (sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)), GPURESIDENTRW);
                }
                else if((sharedContext.resizedTargetImage.firstElement() instanceof FloatType))
                {
                    if(!usesFloatGPU)
                    {
                        // Conversion buffers are only needed if the representation is double later on
                        conversionEntryBuffer = context.createFloatBuffer((int) (sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)), GPURESIDENTRW);
                    }
                }
                else if((sharedContext.resizedTargetImage.firstElement() instanceof DoubleType))
                {
                    // Due to the pretest this makes usesFloatGPU = false => conversion buffers are not needed
                    // just keeping this to keep the else throw clause
                }
                else
                {
                    throw new RuntimeException("Illegal image type");
                }
            }
            if(sharedContext.resizedTargetImage.firstElement() instanceof ByteType)
            {
                conversionProgramKernel.rewind();
                // !NOT! the resized size for the conversion buffer
                conversionProgramKernel.putArg(conversionEntryBuffer).putArg(entryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                deConversionProgramKernel.rewind();
                deConversionProgramKernel.putArg(fullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)));
            }
            if(sharedContext.resizedTargetImage.firstElement() instanceof UnsignedByteType)
            {
                conversionProgramKernel.rewind();
                // !NOT! the resized size for the conversion buffer
                conversionProgramKernel.putArg(conversionEntryBuffer).putArg(entryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                deConversionProgramKernel.rewind();
                deConversionProgramKernel.putArg(fullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)));
            }
            else if((sharedContext.resizedTargetImage.firstElement() instanceof ShortType))
            {
                conversionProgramKernel.rewind();
                // !NOT! the resized size for the conversion buffer
                conversionProgramKernel.putArg(conversionEntryBuffer).putArg(entryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                deConversionProgramKernel.rewind();
                deConversionProgramKernel.putArg(fullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)));
            }
            else if((sharedContext.resizedTargetImage.firstElement() instanceof UnsignedShortType))
            {
                conversionProgramKernel.rewind();
                // !NOT! the resized size for the conversion buffer
                conversionProgramKernel.putArg(conversionEntryBuffer).putArg(entryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                deConversionProgramKernel.rewind();
                deConversionProgramKernel.putArg(fullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)));
            }
            else if((sharedContext.resizedTargetImage.firstElement() instanceof IntType))
            {
                conversionProgramKernel.rewind();
                // !NOT! the resized size for the conversion buffer
                conversionProgramKernel.putArg(conversionEntryBuffer).putArg(entryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                deConversionProgramKernel.rewind();
                deConversionProgramKernel.putArg(fullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)));
            }
            else if((sharedContext.resizedTargetImage.firstElement() instanceof UnsignedIntType))
            {
                conversionProgramKernel.rewind();
                // !NOT! the resized size for the conversion buffer
                conversionProgramKernel.putArg(conversionEntryBuffer).putArg(entryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                deConversionProgramKernel.rewind();
                deConversionProgramKernel.putArg(fullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)));
            }
            else if((sharedContext.resizedTargetImage.firstElement() instanceof LongType))
            {
                conversionProgramKernel.rewind();
                // !NOT! the resized size for the conversion buffer
                conversionProgramKernel.putArg(conversionEntryBuffer).putArg(entryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                deConversionProgramKernel.rewind();
                deConversionProgramKernel.putArg(fullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)));
            }
            else if((sharedContext.resizedTargetImage.firstElement() instanceof UnsignedLongType))
            {
                conversionProgramKernel.rewind();
                // !NOT! the resized size for the conversion buffer
                conversionProgramKernel.putArg(conversionEntryBuffer).putArg(entryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                deConversionProgramKernel.rewind();
                deConversionProgramKernel.putArg(fullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)));
            }
            else if((sharedContext.resizedTargetImage.firstElement() instanceof FloatType))
            {
                conversionProgramKernel.rewind();
                // !NOT! the resized size for the conversion buffer
                conversionProgramKernel.putArg(conversionEntryBuffer).putArg(entryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                if(!usesFloatGPU)
                {
                    // Conversion buffers are only needed if the representation is double later on
                    conversionProgramKernel.rewind();
                    // !NOT! the resized size for the conversion buffer
                    conversionProgramKernel.putArg(conversionEntryBuffer).putArg(entryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                    deConversionProgramKernel.rewind();
                    deConversionProgramKernel.putArg(fullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)));
                }
            }
        }
        
        private void allocateMemory()
        {
            // allocate the pyramid memory
            sourcePyramid = new OCLSourcePyramidSlice[pyramidDepth];
            targetPyramid = new OCLTargetPyramidSlice[pyramidDepth];
            parallelSumReductionBuffers = new CLBuffer[2];// 2 parallel reduction buffers are needed
            {
                long width = sharedContext.img.dimension(0);
                long height = sharedContext.img.dimension(1);
                if(width*height > Integer.MAX_VALUE)
                {
                    throw new RuntimeException("Cannot allocate more than " + Integer.MAX_VALUE);
                }
                if(usesFloatGPU)
                {
                    switch(sharedContext.transformationType) {
                    case TRANSLATION:
                        maskBuffer = context.createFloatBuffer((int)(width*height), GPURESIDENTRW);
                        //maskBuffer.getCLSize();
                        gradient0 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient0.getCLSize();
                        gradient1 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient1.getCLSize();
                        hessian00 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian00.getCLSize();
                        hessian01 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian01.getCLSize();
                        hessian11 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian11.getCLSize();
                        for(int l = 0;l < 2;l++)
                        {
                            parallelSumReductionBuffers[l] = context.createFloatBuffer(maximumSumReductionBlockNr, GPURESIDENTRW);
                            //parallelSumReductionBuffers[l].getCLSize();
                        }
                        break;
                    case RIGIDBODY:
                        maskBuffer = context.createFloatBuffer((int)(width*height), GPURESIDENTRW);
                        //maskBuffer.getCLSize();
                        gradient0 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient0.getCLSize();
                        gradient1 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient1.getCLSize();
                        gradient2 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient2.getCLSize();
                        hessian00 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian00.getCLSize();
                        hessian01 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian01.getCLSize();
                        hessian02 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian02.getCLSize();
                        hessian11 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian11.getCLSize();
                        hessian12 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian12.getCLSize();
                        hessian22 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian22.getCLSize();
                        for(int l = 0;l < 2;l++)
                        {
                            parallelSumReductionBuffers[l] = context.createFloatBuffer(maximumSumReductionBlockNr, GPURESIDENTRW);
                            //parallelSumReductionBuffers[l].getCLSize();
                        }
                        break;
                    case SCALEDROTATION:
                        maskBuffer = context.createFloatBuffer((int)(width*height), GPURESIDENTRW);
                        //maskBuffer.getCLSize();
                        gradient0 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient0.getCLSize();
                        gradient1 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient1.getCLSize();
                        gradient2 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient2.getCLSize();
                        gradient3 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient3.getCLSize();
                        hessian00 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian00.getCLSize();
                        hessian01 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian01.getCLSize();
                        hessian02 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian02.getCLSize();
                        hessian03 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian03.getCLSize();
                        hessian11 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian11.getCLSize();
                        hessian12 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian12.getCLSize();
                        hessian13 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian13.getCLSize();
                        hessian22 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian22.getCLSize();
                        hessian23 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian23.getCLSize();
                        hessian33 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian33.getCLSize();
                        for(int l = 0;l < 2;l++)
                        {
                            parallelSumReductionBuffers[l] = context.createFloatBuffer(maximumSumReductionBlockNr, GPURESIDENTRW);
                            //parallelSumReductionBuffers[l].getCLSize();
                        }
                        break;
                    case AFFINE:
                    	/*
                    	 * On the small GPU setup I tested this on the large number of buffers seems
                    	 * to break the compilation of the kernel so I will use flattened buffers
                    	 * and sub-buffers instead. I will only combine two buffers, but this will
                    	 * limit the maximum image size that can be processed on the GPU because of 
                    	 * the maximum buffer size.
                    	 */
                        maskBuffer = context.createFloatBuffer((int)(width*height), GPURESIDENTRW);
                        //maskBuffer.getCLSize();
                        gradient0 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient0.getCLSize();
                        
                        gradient1 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient1.getCLSize();
                        gradient2 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient2.getCLSize();
                    	
                        gradient3 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient3.getCLSize();
                        gradient4 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient4.getCLSize();
                    	
                        gradient5 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient5.getCLSize();
                        hessian00 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian00.getCLSize();
                    	
                        hessian01 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian01.getCLSize();
                        hessian02 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian02.getCLSize();
                    	
                        hessian03 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian03.getCLSize();
                        hessian04 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian04.getCLSize();
                    	
                        hessian05 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian05.getCLSize();
                        hessian11 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian11.getCLSize();
                    	
                        hessian12 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian12.getCLSize();
                        hessian13 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian13.getCLSize();
                    	
                        hessian14 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian14.getCLSize();
                        hessian15 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian15.getCLSize();
                    	
                        hessian22 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian22.getCLSize();
                        hessian23 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian23.getCLSize();
                    	
                        hessian24 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian24.getCLSize();
                        hessian25 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian25.getCLSize();
                    	
                        hessian33 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian33.getCLSize();
                        hessian34 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian34.getCLSize();
                    	
                        hessian35 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian35.getCLSize();
                        hessian44 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian44.getCLSize();
                    	
                        hessian45 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian45.getCLSize();
                        hessian55 = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian55.getCLSize();
                    	
                        for(int l = 0;l < 2;l++)
                        {
                            parallelSumReductionBuffers[l] = context.createFloatBuffer(maximumSumReductionBlockNr, GPURESIDENTRW);
                            //parallelSumReductionBuffers[l].getCLSize();
                        }
                        break;
                    }
                }
                else
                {
                    switch(sharedContext.transformationType) {
                    case TRANSLATION:
                        maskBuffer = context.createDoubleBuffer((int)(width*height), GPURESIDENTRW);
                        //maskBuffer.getCLSize();
                        gradient0 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient0.getCLSize();
                        gradient1 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient1.getCLSize();
                        hessian00 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian00.getCLSize();
                        hessian01 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian01.getCLSize();
                        hessian11 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian11.getCLSize();
                        for(int l = 0;l < 2;l++)
                        {
                            parallelSumReductionBuffers[l] = context.createDoubleBuffer(maximumSumReductionBlockNr, GPURESIDENTRW);
                            //parallelSumReductionBuffers[l].getCLSize();
                        }
                        break;
                    case RIGIDBODY:
                        maskBuffer = context.createDoubleBuffer((int)(width*height), GPURESIDENTRW);
                        //maskBuffer.getCLSize(); //TODO: Remove these, they were for debugging
                        gradient0 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient0.getCLSize();
                        gradient1 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient1.getCLSize();
                        gradient2 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient2.getCLSize();
                        hessian00 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian00.getCLSize();
                        hessian01 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian01.getCLSize();
                        hessian02 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian02.getCLSize();
                        hessian11 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian11.getCLSize();
                        hessian12 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian12.getCLSize();
                        hessian22 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian22.getCLSize();
                        for(int l = 0;l < 2;l++)
                        {
                            parallelSumReductionBuffers[l] = context.createDoubleBuffer(maximumSumReductionBlockNr, GPURESIDENTRW);
                            //parallelSumReductionBuffers[l].getCLSize();
                        }
                        break;
                    case SCALEDROTATION:
                        maskBuffer = context.createDoubleBuffer((int)(width*height), GPURESIDENTRW);
                        //maskBuffer.getCLSize();
                        gradient0 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient0.getCLSize();
                        gradient1 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient1.getCLSize();
                        gradient2 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient2.getCLSize();
                        gradient3 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient3.getCLSize();
                        hessian00 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian00.getCLSize();
                        hessian01 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian01.getCLSize();
                        hessian02 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian02.getCLSize();
                        hessian03 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian03.getCLSize();
                        hessian11 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian11.getCLSize();
                        hessian12 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian12.getCLSize();
                        hessian13 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian13.getCLSize();
                        hessian22 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian22.getCLSize();
                        hessian23 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian23.getCLSize();
                        hessian33 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian33.getCLSize();
                        for(int l = 0;l < 2;l++)
                        {
                            parallelSumReductionBuffers[l] = context.createDoubleBuffer(maximumSumReductionBlockNr, GPURESIDENTRW);
                            //parallelSumReductionBuffers[l].getCLSize();
                        }
                        break;
                    case AFFINE:
                    	/*
                    	 * On the small GPU setup I tested this on the large number of buffers seems
                    	 * to break the compilation of the kernel so I will use flattened buffers
                    	 * and sub-buffers instead. I will only combine two buffers, but this will
                    	 * limit the maximum image size that can be processed on the GPU because of 
                    	 * the maximum buffer size.
                    	 */                    	
                        maskBuffer = context.createDoubleBuffer((int)(width*height), GPURESIDENTRW);
                        //maskBuffer.getCLSize();
                        gradient0 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient0.getCLSize();
                    	
                        gradient1 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient1.getCLSize();
                        gradient2 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient2.getCLSize();
                    	
                        gradient3 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient3.getCLSize();
                        gradient4 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient4.getCLSize();
                    	
                        gradient5 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //gradient5.getCLSize();
                        hessian00 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian00.getCLSize();
                    	
                        hessian01 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian01.getCLSize();
                        hessian02 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian02.getCLSize();
                    	
                        hessian03 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian03.getCLSize();
                        hessian04 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian04.getCLSize();
                    	
                        hessian05 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian05.getCLSize();
                        hessian11 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian11.getCLSize();
                    	
                        hessian12 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian12.getCLSize();
                        hessian13 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian13.getCLSize();
                    	
                        hessian14 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian14.getCLSize();
                        hessian15 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian15.getCLSize();
                    	
                        hessian22 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian22.getCLSize();
                        hessian23 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian23.getCLSize();
                    	
                        hessian24 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian24.getCLSize();
                        hessian25 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian25.getCLSize();
                    	
                        hessian33 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian33.getCLSize();
                        hessian34 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian34.getCLSize();
                    	
                        hessian35 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian35.getCLSize();
                        hessian44 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian44.getCLSize();
                    	
                        hessian45 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian45.getCLSize();
                        hessian55 = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        //hessian55.getCLSize();
                    	
                        for(int l = 0;l < 2;l++)
                        {
                            parallelSumReductionBuffers[l] = context.createDoubleBuffer(maximumSumReductionBlockNr, GPURESIDENTRW);
                            //parallelSumReductionBuffers[l].getCLSize();
                        }
                        break;
                    }
                }
                for(int j = 0;j < pyramidDepth; j++)
                {
                    sourcePyramid[j] = new OCLSourcePyramidSlice();
                    targetPyramid[j] = new OCLTargetPyramidSlice();
                    sourcePyramid[j].width = width;
                    sourcePyramid[j].height = height;
                    targetPyramid[j].width = width;
                    targetPyramid[j].height = height;   
                    if(usesFloatGPU)
                    {
                        sourcePyramid[j].Image = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        sourcePyramid[j].xGradient = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        sourcePyramid[j].yGradient= context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                        targetPyramid[j].Coefficient = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                    }
                    else
                    {
                        sourcePyramid[j].Image = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        sourcePyramid[j].xGradient = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        sourcePyramid[j].yGradient= context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                        targetPyramid[j].Coefficient = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);   
                    }
                    //sourcePyramid[j].Image.getCLSize();
                    //sourcePyramid[j].xGradient.getCLSize();
                    //sourcePyramid[j].yGradient.getCLSize();
                    //targetPyramid[j].Coefficient.getCLSize();
                    width /= 2;
                    height /= 2;
                }
            }
            // Really ugly code but I couldn't figure out how to do this more elegantly
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
            else if((sharedContext.img.firstElement() instanceof LongType)||(sharedContext.img.firstElement() instanceof UnsignedLongType))
            {
                conversionEntryBuffer = context.createLongBuffer((int) (sharedContext.img.dimension(0)*sharedContext.img.dimension(1)), GPURESIDENTRW);
            }
            else if((sharedContext.img.firstElement() instanceof FloatType))
            {
                if(!usesFloatGPU)
                {
                    // Conversion buffers are only needed if the representation is double later on
                    conversionEntryBuffer = context.createFloatBuffer((int) (sharedContext.img.dimension(0)*sharedContext.img.dimension(1)), GPURESIDENTRW);
                }
            }
            else if((sharedContext.img.firstElement() instanceof DoubleType))
            {
                // Due to the pretest this makes usesFloatGPU = false => conversion buffers are not needed
                // just keeping this to keep the else throw clause
            }
            else
            {
                throw new RuntimeException("Illegal image type");
            }
            if(usesFloatGPU)
            {
                entryImageBuffer = context.createFloatBuffer((int) (sharedContext.img.dimension(0)*sharedContext.img.dimension(1)), GPURESIDENTRW);
                fullSizedGPUResidentHelperBuffer = context.createFloatBuffer((int) (sharedContext.img.dimension(0)*sharedContext.img.dimension(1)), GPURESIDENTRW);
                secondaryGPUResidentHelperBuffer = context.createFloatBuffer(((((int)sharedContext.img.dimension(0)))*(((int)sharedContext.img.dimension(1)))), GPURESIDENTRW);
            }
            else
            {
                entryImageBuffer = context.createDoubleBuffer((int) (sharedContext.img.dimension(0)*sharedContext.img.dimension(1)), GPURESIDENTRW);
                fullSizedGPUResidentHelperBuffer = context.createDoubleBuffer((int) (sharedContext.img.dimension(0)*sharedContext.img.dimension(1)), GPURESIDENTRW);
                secondaryGPUResidentHelperBuffer = context.createDoubleBuffer(((((int)sharedContext.img.dimension(0)))*(((int)sharedContext.img.dimension(1)))), GPURESIDENTRW);
            }
        }
        
        private void CompileAndSetupOpenCLKernerls() throws IOException
        {
            uniformBSplineTransformProgramKernels = new CLKernel[NR_OF_OPENCL_KERNELS];
            optimalMultiples = new long[NR_OF_OPENCL_KERNELS];
            if(usesFloatGPU)
            {
            	switch(sharedContext.transformationType) {
            	case TRANSLATION:
            		uniformBSplineTransformProgram = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/UniformBSplineTransform.cl")).build("-D TRANSLATION", device);
            		break;
            	case RIGIDBODY:
            		uniformBSplineTransformProgram = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/UniformBSplineTransform.cl")).build("-D RIGIDBODY", device);
            		break;
				case SCALEDROTATION:
					uniformBSplineTransformProgram = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/UniformBSplineTransform.cl")).build("-D SCALEDROTATION", device);
					break;
				case AFFINE:
					uniformBSplineTransformProgram = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/UniformBSplineTransformAffine.cl")).build("-D AFFINE", device);
					break;
            	}
            }
            else
            {
            	switch(sharedContext.transformationType) {
            	case TRANSLATION:
            		uniformBSplineTransformProgram = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/UniformBSplineTransform.cl")).build("-D TRANSLATION -D USE_DOUBLE", device);
            		break;
            	case RIGIDBODY:
            		uniformBSplineTransformProgram = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/UniformBSplineTransform.cl")).build("-D RIGIDBODY -D USE_DOUBLE", device);
            		break;
				case SCALEDROTATION:
					uniformBSplineTransformProgram = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/UniformBSplineTransform.cl")).build("-D SCALEDROTATION -D USE_DOUBLE", device);
					break;
				case AFFINE:
					uniformBSplineTransformProgram = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/UniformBSplineTransformAffine.cl")).build("-D AFFINE -D USE_DOUBLE", device);
					/*
					 * TODO:
					 * Potentially use SPIR-V precompiled programs
					 * use the following command to compile the OpenCL C source to SPIR-V binary:
					 * clang -target spirv64 -cl-std=CL1.2 -D USE_DOUBLE -O3 -c UniformBSplineTransformAffineExtended.cl -o UniformBSplineTransformAffineExtended_double.spv
					 *
					try (InputStream inpStream = getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/UniformBSplineTransformAffineExtended.cl")) {
					 
						
						/*Map<CLDevice, byte[]> binaries = new HashMap<>();
						if (inpStream == null) {
							throw new IOException("Could not find kernel source for UniformBSplineTransformAffineExtended.cl");
						}
		            	//byte[] spvBytes = inpStream.readAllBytes(); only exists in Java 9 and later, so we do it manually for Java 8 compatibility
		            	ByteArrayOutputStream buffer = new ByteArrayOutputStream();
		            	int nRead;
		            	byte[] data = new byte[16384];
		            	while ((nRead = inpStream.read(data, 0, data.length)) != -1) {
						    buffer.write(data, 0, nRead);
						}
		            	byte[] spvBytes = buffer.toByteArray();
		            	inpStream.close();
		            	binaries.put(device, spvBytes);
		            	uniformBSplineTransformProgram = context.createProgram(binaries).build("-x spirv", device);*/
					//}
					break;
            	}
            }
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp] = uniformBSplineTransformProgram.createCLKernel("CubicBSplinePrefilter2Dpremulhp");
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp] = uniformBSplineTransformProgram.createCLKernel("CubicBSplinePrefilter2DXhp");
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp] = uniformBSplineTransformProgram.createCLKernel("CubicBSplinePrefilter2DYhp");
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhp] = uniformBSplineTransformProgram.createCLKernel("BasicToCardinal2DXhp");
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhp] = uniformBSplineTransformProgram.createCLKernel("BasicToCardinal2DYhp");
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp] = uniformBSplineTransformProgram.createCLKernel("CubicBSplinePrefilter2DDeg7premulhp");
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXDeg7hp] = uniformBSplineTransformProgram.createCLKernel("CubicBSplinePrefilter2DXDeg7hp");
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYDeg7hp] = uniformBSplineTransformProgram.createCLKernel("CubicBSplinePrefilter2DYDeg7hp");
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhpDeg7] = uniformBSplineTransformProgram.createCLKernel("BasicToCardinal2DXhpDeg7");
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhpDeg7] = uniformBSplineTransformProgram.createCLKernel("BasicToCardinal2DYhpDeg7");
            uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX] = uniformBSplineTransformProgram.createCLKernel("reduceDual1DX");
            uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DY] = uniformBSplineTransformProgram.createCLKernel("reduceDual1DY");
            uniformBSplineTransformProgramKernels[KERNEL_antiSymmetricFirMirrorOffBounds1DX] = uniformBSplineTransformProgram.createCLKernel("antiSymmetricFirMirrorOffBounds1DX");
            uniformBSplineTransformProgramKernels[KERNEL_antiSymmetricFirMirrorOffBounds1DY] = uniformBSplineTransformProgram.createCLKernel("antiSymmetricFirMirrorOffBounds1DY");
            uniformBSplineTransformProgramKernels[KERNEL_TargetedCubicBSplinePrefilter2Dpremulhp] = uniformBSplineTransformProgram.createCLKernel("TargetedCubicBSplinePrefilter2Dpremulhp");
            uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory] = uniformBSplineTransformProgram.createCLKernel("sumInLocalMemory");
            uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction] = uniformBSplineTransformProgram.createCLKernel("parallelGroupedSumReduction");            
            
            switch(sharedContext.transformationType) {
            case TRANSLATION:
                uniformBSplineTransformProgramKernels[KERNEL_translationError] = uniformBSplineTransformProgram.createCLKernel("translationError");
                uniformBSplineTransformProgramKernels[KERNEL_translationErrorWithGradAndHess] = uniformBSplineTransformProgram.createCLKernel("translationErrorWithGradAndHess");
                uniformBSplineTransformProgramKernels[KERNEL_translationErrorWithGradAndHessBrent] = uniformBSplineTransformProgram.createCLKernel("translationErrorWithGradAndHessBrent");
                uniformBSplineTransformProgramKernels[KERNEL_translationTransformImageWithBsplineInterpolation] = uniformBSplineTransformProgram.createCLKernel("translationTransformImageWithBsplineInterpolation");
                uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation] = uniformBSplineTransformProgram.createCLKernel("resizingTranslationTransformImageWithBsplineInterpolation");
                uniformBSplineTransformProgramKernels[KERNEL_translationSumInLocalMemoryCombined] = uniformBSplineTransformProgram.createCLKernel("translationSumInLocalMemoryCombined");
                break;
            case RIGIDBODY:
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyError] = uniformBSplineTransformProgram.createCLKernel("rigidBodyError");
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHess] = uniformBSplineTransformProgram.createCLKernel("rigidBodyErrorWithGradAndHess");
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent] = uniformBSplineTransformProgram.createCLKernel("rigidBodyErrorWithGradAndHessBrent");
                uniformBSplineTransformProgramKernels[KERNEL_transformImageWithBsplineInterpolation] = uniformBSplineTransformProgram.createCLKernel("transformImageWithBsplineInterpolation");
                uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation] = uniformBSplineTransformProgram.createCLKernel("resizingTransformImageWithBsplineInterpolation");
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemoryCombined] = uniformBSplineTransformProgram.createCLKernel("sumInLocalMemoryCombined");
                break;
            case SCALEDROTATION:
                uniformBSplineTransformProgramKernels[KERNEL_scaledRotationError] = uniformBSplineTransformProgram.createCLKernel("scaledRotationError");
                uniformBSplineTransformProgramKernels[KERNEL_scaledRotationErrorWithGradAndHess] = uniformBSplineTransformProgram.createCLKernel("scaledRotationErrorWithGradAndHess");
                uniformBSplineTransformProgramKernels[KERNEL_scaledRotationErrorWithGradAndHessBrent] = uniformBSplineTransformProgram.createCLKernel("scaledRotationErrorWithGradAndHessBrent");
                uniformBSplineTransformProgramKernels[KERNEL_scaledRotationTransformImageWithBsplineInterpolation] = uniformBSplineTransformProgram.createCLKernel("scaledRotationTransformImageWithBsplineInterpolation");
                uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation] = uniformBSplineTransformProgram.createCLKernel("resizingScaledRotationTransformImageWithBsplineInterpolation");
                uniformBSplineTransformProgramKernels[KERNEL_scaledRotationSumInLocalMemoryCombined] = uniformBSplineTransformProgram.createCLKernel("sumInLocalMemoryScaledRotationCombined");
                break;
            case AFFINE:
                uniformBSplineTransformProgramKernels[KERNEL_affineError] = uniformBSplineTransformProgram.createCLKernel("affineError");
                uniformBSplineTransformProgramKernels[KERNEL_affineErrorWithGradAndHess] = uniformBSplineTransformProgram.createCLKernel("affineErrorWithGradAndHess");
                uniformBSplineTransformProgramKernels[KERNEL_affineErrorWithGradAndHessBrent] = uniformBSplineTransformProgram.createCLKernel("affineErrorWithGradAndHessBrent");
                uniformBSplineTransformProgramKernels[KERNEL_affineTransformImageWithBsplineInterpolation] = uniformBSplineTransformProgram.createCLKernel("affineTransformImageWithBsplineInterpolation");
                uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation] = uniformBSplineTransformProgram.createCLKernel("resizingAffineTransformImageWithBsplineInterpolation");
                uniformBSplineTransformProgramKernels[KERNEL_affineSumInLocalMemoryCombined] = uniformBSplineTransformProgram.createCLKernel("sumInLocalMemoryAffineCombined");
                break;
            }
            
            /*
            The following code determines the optimal multiple size for each kernel.
            This may be different for each kernel (usually not for NVidia GPU's where
            this is either 32 or 64 but for example on Intel GPU where unsynchronized
            kernels may be run on multiple compute devices (SIMD processors) 
            concurrently.
            */
            optimalMultiples[KERNEL_CubicBSplinePrefilter2Dpremulhp] = uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].getPreferredWorkGroupSizeMultiple(device);
            optimalMultiples[KERNEL_CubicBSplinePrefilter2DXhp] = uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp].getPreferredWorkGroupSizeMultiple(device);
            optimalMultiples[KERNEL_CubicBSplinePrefilter2DYhp] = uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp].getPreferredWorkGroupSizeMultiple(device);
            optimalMultiples[KERNEL_BasicToCardinal2DXhp] = uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhp].getPreferredWorkGroupSizeMultiple(device);
            optimalMultiples[KERNEL_BasicToCardinal2DYhp] = uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhp].getPreferredWorkGroupSizeMultiple(device);
            optimalMultiples[KERNEL_CubicBSplinePrefilter2DDeg7premulhp] = uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp].getPreferredWorkGroupSizeMultiple(device);
            optimalMultiples[KERNEL_CubicBSplinePrefilter2DXDeg7hp] = uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXDeg7hp].getPreferredWorkGroupSizeMultiple(device);
            optimalMultiples[KERNEL_CubicBSplinePrefilter2DYDeg7hp] = uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYDeg7hp].getPreferredWorkGroupSizeMultiple(device);
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
                optimalMultiples[KERNEL_translationError] = uniformBSplineTransformProgramKernels[KERNEL_translationError].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_translationErrorWithGradAndHess] = uniformBSplineTransformProgramKernels[KERNEL_translationErrorWithGradAndHess].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_translationErrorWithGradAndHessBrent] = uniformBSplineTransformProgramKernels[KERNEL_translationErrorWithGradAndHessBrent].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_translationTransformImageWithBsplineInterpolation] = uniformBSplineTransformProgramKernels[KERNEL_translationTransformImageWithBsplineInterpolation].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_resizeTransformImageWithBsplineInterpolation] = uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_translationSumInLocalMemoryCombined] = uniformBSplineTransformProgramKernels[KERNEL_translationSumInLocalMemoryCombined].getPreferredWorkGroupSizeMultiple(device);
                
                blockSizesParallel = (int)Math.min(Math.min(uniformBSplineTransformProgramKernels[KERNEL_translationErrorWithGradAndHessBrent].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_translationErrorWithGradAndHessBrent].getLocalMemorySize(device))/((usesFloatGPU?4:8)*7/*7 buffers are needed*/)), blocksizeMultiplier*optimalMultiples[KERNEL_translationErrorWithGradAndHessBrent]);
                blockSizesParallel -= (blockSizesParallel % optimalMultiples[KERNEL_translationErrorWithGradAndHessBrent]);
                if(blockSizesParallel == 0)
                {
                    blockSizesParallel = 1;// Fallback solution minimum is 1
                }
                
                maximumElementsForLocalFPTcombinedSum = (int) Math.min(uniformBSplineTransformProgramKernels[KERNEL_translationSumInLocalMemoryCombined].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_translationSumInLocalMemoryCombined].getLocalMemorySize(device))/(usesFloatGPU?4:8));
                break;
            case RIGIDBODY:
                optimalMultiples[KERNEL_rigidBodyError] = uniformBSplineTransformProgramKernels[KERNEL_rigidBodyError].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_rigidBodyErrorWithGradAndHess] = uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHess].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_rigidBodyErrorWithGradAndHessBrent] = uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_transformImageWithBsplineInterpolation] = uniformBSplineTransformProgramKernels[KERNEL_transformImageWithBsplineInterpolation].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_resizeTransformImageWithBsplineInterpolation] = uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_sumInLocalMemoryCombined] = uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemoryCombined].getPreferredWorkGroupSizeMultiple(device);
                
                blockSizesParallel = (int)Math.min(Math.min(uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent].getLocalMemorySize(device))/((usesFloatGPU?4:8)*11/*11 buffers are needed*/)), blocksizeMultiplier*optimalMultiples[KERNEL_rigidBodyErrorWithGradAndHessBrent]);
                blockSizesParallel -= (blockSizesParallel % optimalMultiples[KERNEL_rigidBodyErrorWithGradAndHessBrent]);
                if(blockSizesParallel == 0)
                {
                    blockSizesParallel = 1;// Fallback solution minimum is 1
                }
                
                maximumElementsForLocalFPTcombinedSum = (int) Math.min(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemoryCombined].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemoryCombined].getLocalMemorySize(device))/(usesFloatGPU?4:8));
                break;
            case SCALEDROTATION:
                optimalMultiples[KERNEL_scaledRotationError] = uniformBSplineTransformProgramKernels[KERNEL_scaledRotationError].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_scaledRotationErrorWithGradAndHess] = uniformBSplineTransformProgramKernels[KERNEL_scaledRotationErrorWithGradAndHess].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_scaledRotationErrorWithGradAndHessBrent] = uniformBSplineTransformProgramKernels[KERNEL_scaledRotationErrorWithGradAndHessBrent].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_scaledRotationTransformImageWithBsplineInterpolation] = uniformBSplineTransformProgramKernels[KERNEL_scaledRotationTransformImageWithBsplineInterpolation].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_resizeTransformImageWithBsplineInterpolation] = uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_scaledRotationSumInLocalMemoryCombined] = uniformBSplineTransformProgramKernels[KERNEL_scaledRotationSumInLocalMemoryCombined].getPreferredWorkGroupSizeMultiple(device);
                
                blockSizesParallel = (int)Math.min(Math.min(uniformBSplineTransformProgramKernels[KERNEL_scaledRotationErrorWithGradAndHessBrent].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_scaledRotationErrorWithGradAndHessBrent].getLocalMemorySize(device))/((usesFloatGPU?4:8)*16/*16 buffers are needed*/)), blocksizeMultiplier*optimalMultiples[KERNEL_scaledRotationErrorWithGradAndHessBrent]);
                blockSizesParallel -= (blockSizesParallel % optimalMultiples[KERNEL_scaledRotationErrorWithGradAndHessBrent]);
                if(blockSizesParallel == 0)
                {
                    blockSizesParallel = 1;// Fallback solution minimum is 1
                }
                
                maximumElementsForLocalFPTcombinedSum = (int) Math.min(uniformBSplineTransformProgramKernels[KERNEL_scaledRotationSumInLocalMemoryCombined].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_scaledRotationSumInLocalMemoryCombined].getLocalMemorySize(device))/(usesFloatGPU?4:8));
                break;
            case AFFINE:
                optimalMultiples[KERNEL_affineError] = uniformBSplineTransformProgramKernels[KERNEL_affineError].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_affineErrorWithGradAndHess] = uniformBSplineTransformProgramKernels[KERNEL_affineErrorWithGradAndHess].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_affineErrorWithGradAndHessBrent] = uniformBSplineTransformProgramKernels[KERNEL_affineErrorWithGradAndHessBrent].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_affineTransformImageWithBsplineInterpolation] = uniformBSplineTransformProgramKernels[KERNEL_affineTransformImageWithBsplineInterpolation].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_resizeTransformImageWithBsplineInterpolation] = uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation].getPreferredWorkGroupSizeMultiple(device);
                optimalMultiples[KERNEL_affineSumInLocalMemoryCombined] = uniformBSplineTransformProgramKernels[KERNEL_affineSumInLocalMemoryCombined].getPreferredWorkGroupSizeMultiple(device);
                
                blockSizesParallel = (int)Math.min(Math.min(uniformBSplineTransformProgramKernels[KERNEL_affineErrorWithGradAndHessBrent].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_affineErrorWithGradAndHessBrent].getLocalMemorySize(device))/((usesFloatGPU?4:8)*29/*29 buffers are needed*/)), blocksizeMultiplier*optimalMultiples[KERNEL_affineErrorWithGradAndHessBrent]);
                blockSizesParallel -= (blockSizesParallel % optimalMultiples[KERNEL_affineErrorWithGradAndHessBrent]);
                if(blockSizesParallel == 0)
                {
                    blockSizesParallel = 1;// Fallback solution minimum is 1
                }
                
                maximumElementsForLocalFPTcombinedSum = (int) Math.min(uniformBSplineTransformProgramKernels[KERNEL_affineSumInLocalMemoryCombined].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_affineSumInLocalMemoryCombined].getLocalMemorySize(device))/(usesFloatGPU?4:8));
                break;
            }
            
            
            maximumElementsForLocalFPTsum = (int) Math.min(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].getLocalMemorySize(device))/(usesFloatGPU?4:8));
            blockSizesFPT = (int)Math.min(Math.min(uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].getLocalMemorySize(device))/(usesFloatGPU?4:8)), blocksizeMultiplier*optimalMultiples[KERNEL_parallelGroupedSumReduction]);
            blockSizesFPT -= (blockSizesFPT % optimalMultiples[KERNEL_parallelGroupedSumReduction]);
            if(blockSizesFPT == 0)
            {
                blockSizesFPT = 1;// Fallback solution minimum is 1
            }
            // Ugly code but I didn't figure out how to do it more elegantly
            if(sharedContext.img.firstElement() instanceof ByteType)
            {
                if(usesFloatGPU)
                {
                    conversionProgram = context.createProgram(System.class.getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/Conversion.cl")).build("-D SDT=char -D CMD=convert_char_sat_rte -D TDT=float -D MAXVAL=\"127.0f\"",device);
                }
                else
                {
                    conversionProgram = context.createProgram(System.class.getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/Conversion.cl")).build("-D SDT=char -D CMD=convert_char_sat_rte -D TDT=double -D MAXVAL=\"127.0\"",device);
                }
                conversionProgramKernel = conversionProgram.createCLKernel("Convert");
                conversionProgramKernel.putArg(conversionEntryBuffer).putArg(entryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                deConversionProgramKernel = conversionProgram.createCLKernel("deConvert");
                deConversionProgramKernel.putArg(fullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
            }
            if(sharedContext.img.firstElement() instanceof UnsignedByteType)
            {
                if(usesFloatGPU)
                {
                    conversionProgram = context.createProgram(System.class.getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/Conversion.cl")).build("-D SDT=uchar -D CMD=convert_uchar_sat_rte -D TDT=float -D MAXVAL=\"255.0f\"",device);
                }
                else
                {
                    conversionProgram = context.createProgram(System.class.getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/Conversion.cl")).build("-D SDT=uchar -D CMD=convert_uchar_sat_rte -D TDT=double -D MAXVAL=\"255.0\"",device);
                }
                conversionProgramKernel = conversionProgram.createCLKernel("Convert");
                conversionProgramKernel.putArg(conversionEntryBuffer).putArg(entryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                deConversionProgramKernel = conversionProgram.createCLKernel("deConvert");
                deConversionProgramKernel.putArg(fullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
            }
            else if((sharedContext.img.firstElement() instanceof ShortType))
            {
                if(usesFloatGPU)
                {
                    conversionProgram = context.createProgram(System.class.getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/Conversion.cl")).build("-D SDT=short -D CMD=convert_short_sat_rte -D TDT=float -D MAXVAL=\"32767.0f\"",device);
                }
                else
                {
                    conversionProgram = context.createProgram(System.class.getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/Conversion.cl")).build("-D SDT=short -D CMD=convert_short_sat_rte -D TDT=double -D MAXVAL=\"32767.0\"",device);
                }
                conversionProgramKernel = conversionProgram.createCLKernel("Convert");
                conversionProgramKernel.putArg(conversionEntryBuffer).putArg(entryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                deConversionProgramKernel = conversionProgram.createCLKernel("deConvert");
                deConversionProgramKernel.putArg(fullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
            }
            else if((sharedContext.img.firstElement() instanceof UnsignedShortType))
            {
                if(usesFloatGPU)
                {
                    conversionProgram = context.createProgram(System.class.getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/Conversion.cl")).build("-D SDT=ushort -D CMD=convert_ushort_sat_rte -D TDT=float -D MAXVAL=\"65535.0f\"",device);
                }
                else
                {
                    conversionProgram = context.createProgram(System.class.getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/Conversion.cl")).build("-D SDT=ushort -D CMD=convert_ushort_sat_rte -D TDT=double -D MAXVAL=\"65535.0\"",device);
                }
                conversionProgramKernel = conversionProgram.createCLKernel("Convert");
                conversionProgramKernel.putArg(conversionEntryBuffer).putArg(entryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                deConversionProgramKernel = conversionProgram.createCLKernel("deConvert");
                deConversionProgramKernel.putArg(fullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
            }
            else if((sharedContext.img.firstElement() instanceof IntType))
            {
                if(usesFloatGPU)
                {
                    conversionProgram = context.createProgram(System.class.getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/Conversion.cl")).build("-D SDT=int -D CMD=convert_int_sat_rte -D TDT=float -D MAXVAL=\"2147483647.0f\"",device);
                }
                else
                {
                    conversionProgram = context.createProgram(System.class.getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/Conversion.cl")).build("-D SDT=int -D CMD=convert_int_sat_rte -D TDT=double -D MAXVAL=\"2147483647.0\"",device);
                }
                conversionProgramKernel = conversionProgram.createCLKernel("Convert");
                conversionProgramKernel.putArg(conversionEntryBuffer).putArg(entryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                deConversionProgramKernel = conversionProgram.createCLKernel("deConvert");
                deConversionProgramKernel.putArg(fullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
            }
            else if((sharedContext.img.firstElement() instanceof UnsignedIntType))
            {
                if(usesFloatGPU)
                {
                    conversionProgram = context.createProgram(System.class.getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/Conversion.cl")).build("-D SDT=uint -D CMD=convert_uint_sat_rte -D TDT=float -D MAXVAL=\"4294967295.0f\"",device);
                }
                else
                {
                    conversionProgram = context.createProgram(System.class.getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/Conversion.cl")).build("-D SDT=uint -D CMD=convert_uint_sat_rte -D TDT=double -D MAXVAL=\"4294967295.0\"",device);
                }
                conversionProgramKernel = conversionProgram.createCLKernel("Convert");
                conversionProgramKernel.putArg(conversionEntryBuffer).putArg(entryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                deConversionProgramKernel = conversionProgram.createCLKernel("deConvert");
                deConversionProgramKernel.putArg(fullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
            }
            else if((sharedContext.img.firstElement() instanceof LongType))
            {
                conversionProgram = context.createProgram(System.class.getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/Conversion.cl")).build("-D SDT=long -D CMD=convert_long_sat_rte -D TDT=double -D MAXVAL=\"9223372036854775807.0\"",device);
                conversionProgramKernel = conversionProgram.createCLKernel("Convert");
                conversionProgramKernel.putArg(conversionEntryBuffer).putArg(entryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                deConversionProgramKernel = conversionProgram.createCLKernel("deConvert");
                deConversionProgramKernel.putArg(fullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
            }
            else if((sharedContext.img.firstElement() instanceof UnsignedLongType))
            {
                conversionProgram = context.createProgram(System.class.getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/Conversion.cl")).build("-D SDT=ulong -D CMD=convert_ulong_sat_rte -D TDT=double -D MAXVAL=\"18446744073709551615.0\"",device);
                conversionProgramKernel = conversionProgram.createCLKernel("Convert");
                conversionProgramKernel.putArg(conversionEntryBuffer).putArg(entryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                deConversionProgramKernel = conversionProgram.createCLKernel("deConvert");
                deConversionProgramKernel.putArg(fullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
            }
            else if((sharedContext.img.firstElement() instanceof FloatType))
            {
                if(!usesFloatGPU)
                {
                    // Conversion buffers are only needed if the representation is double later on
                    conversionProgram = context.createProgram(System.class.getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/Conversion.cl")).build("-D SDT=float -D CMD=convert_float_sat_rte -D TDT=double",device);
                    conversionProgramKernel = conversionProgram.createCLKernel("Convert");
                    conversionProgramKernel.putArg(conversionEntryBuffer).putArg(entryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                    deConversionProgramKernel = conversionProgram.createCLKernel("deConvert");
                    deConversionProgramKernel.putArg(fullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                }
            }
            // otherwise nothing has to be done
            
        }

        @Override
        public void run() {
            while(!sharedContext.getNextAlignmentTarget(scat))
            {                
                putTargetImageIntoPipelineEntry();
                // Note that the image is (also converted if necessary) in entryImageBuffers[id], the pyramid can now be constructed
                constructTargetImagePyramid();
                // Now put the source image into the pipeline (entryImageBuffer[id]) and convert it if necessary
                putSourceImageIntoPipelineEntry();
                // Construct the image and derivative pyramids
                constructSourceImagePyramid();
                // Now that we have the pyramid the optimization can commence
                doRegistration();
                switch(sharedContext.transformationType)
                {
                    case TRANSLATION:
                        ((TranslationTransformation)scat.transformation).offsetx = offsetx;
                        ((TranslationTransformation)scat.transformation).offsety = offsety;
                        // reset the transformation values for the next image
                        offsetx = 0.0;
                        offsety = 0.0;
                        break;
                    case RIGIDBODY:
                        ((RigidBodyTransformation)scat.transformation).angle = angle;
                        ((RigidBodyTransformation)scat.transformation).offsetx = offsetx;
                        ((RigidBodyTransformation)scat.transformation).offsety = offsety;
                        // reset the transformation values for the next image
                        offsetx = 0.0;
                        offsety = 0.0;
                        angle = 0.0;
                        break;
                    case SCALEDROTATION:
                        ((ScaledRotationTransformation)scat.transformation).angle = angle;
                        ((ScaledRotationTransformation)scat.transformation).scale = scale;
                        ((ScaledRotationTransformation)scat.transformation).offsetx = offsetx;
                        ((ScaledRotationTransformation)scat.transformation).offsety = offsety;
                        // reset the transformation values for the next image
                        offsetx = 0.0;
                        offsety = 0.0;
                        angle = 0.0;
                        scale = 1.0;
                        break;
                    case AFFINE:
                        ((AffineTransformation)scat.transformation).a11 = a11;
                        ((AffineTransformation)scat.transformation).a12 = a12;
                        ((AffineTransformation)scat.transformation).a21 = a21;
                        ((AffineTransformation)scat.transformation).a22 = a22;
                        ((AffineTransformation)scat.transformation).offsetx = offsetx;
                        ((AffineTransformation)scat.transformation).offsety = offsety;
                        // reset the transformation values for the next image
                        offsetx = 0.0;
                        offsety = 0.0;
                        a11 = a22 = 1.0;
                        a12 = a21 = 0.0;
                        break;
                }
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
                    // Now put the source image into the pipeline (entryImageBuffer[id]) and convert it if necessary
                    putSourceImageIntoPipelineEntry();
                    // Construct the image and derivative pyramids
                    constructSourceImagePyramid();
                    // Now that we have the pyramid the optimization can commence
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
                            ((ScaledRotationTransformation)scat.transformation).angle = angle;
                            ((ScaledRotationTransformation)scat.transformation).scale = scale;
                            ((ScaledRotationTransformation)scat.transformation).offsetx = offsetx;
                            ((ScaledRotationTransformation)scat.transformation).offsety = offsety;
                            // reset the transformation values for the next image
                            offsetx = 0.0;
                            offsety = 0.0;
                            angle = 0.0;
                            scale = 1.0;
                            break;
                        case AFFINE:
                            ((AffineTransformation)scat.transformation).a11 = a11;
                            ((AffineTransformation)scat.transformation).a12 = a12;
                            ((AffineTransformation)scat.transformation).a21 = a21;
                            ((AffineTransformation)scat.transformation).a22 = a22;
                            ((AffineTransformation)scat.transformation).offsetx = offsetx;
                            ((AffineTransformation)scat.transformation).offsety = offsety;
                            // reset the transformation values for the next image
                            offsetx = 0.0;
                            offsety = 0.0;
                            a11 = a22 = 1.0;
                            a12 = a21 = 0.0;
                            break;
                    }
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
            // The following code requires the current position to have been reset to the 0vector
            if(sharedContext.getResizeAfterRegistration())
            {
                /*
                 * This means the target buffer has to be resized and we can get rid of the remaining buffers
                 * entryImageBuffer <- source
                 * fullSizedGPUResidentHelperBuffer <- target
                 * then the deconversion
                 * fullSizedGPUResidentHelperBuffer <- source
                 * conversionEntryBuffer <- target
                 */
                resizeAllocatedBuffers();
            }
            
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
                        transformImageScaledRotation();
                        break;
                    case AFFINE:
                        transformImageAffine();
                        break;
                }
                queue.finish();
            }
            
            cleanup();
        }
        
        private void transformImageTranslation()
        {
            int width = (int)sharedContext.img.dimension(0);
            int height = (int)sharedContext.img.dimension(1);
            int localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].getWorkGroupSize(device);  // Local work size dimensions
            int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2Dpremulhp], width*height);   // rounded up to the nearest multiple of the localWorkSize
            // pre-multiply the image
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].putArg(entryImageBuffer).putArg(width*height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp],0,globalWorkSize,localWorkSize);
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp].getWorkGroupSize(device);
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp].putArg(entryImageBuffer).putArg(width).putArg(height);
            // Conversion to B-spline coefficients along X axis (Group size must be >= height)
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DXhp], height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp].rewind();
            // Now along Y-axis
            // Has to be pre-multiplied by lambda again!!!
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2Dpremulhp], width*height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp].putArg(entryImageBuffer).putArg(width).putArg(height);
            // Conversion to B-spline coefficients along Y axis (Group size must be >= width)
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DYhp], width);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp].rewind();
            
            if(sharedContext.getResizeAfterRegistration())
            {
                int targetwidth = (int)sharedContext.resizedTargetImage.dimension(0);
                int targetheight = (int)sharedContext.resizedTargetImage.dimension(1);
                /*
                Now the B-spline coefficients are in the entryImageBuffer,
                lastly the image needs to be transformed, rescaled back and then 
                convert it back to the original image format.
                */
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation].getWorkGroupSize(device);  // Local work size dimensions
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_resizeTransformImageWithBsplineInterpolation], targetwidth*targetheight);   // rounded up to the nearest multiple of the localWorkSize
                if(usesFloatGPU)
                {
                    uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation]
                            .putArg(entryImageBuffer)
                            .putArg(fullSizedGPUResidentHelperBuffer)
                            .putArg(width)
                            .putArg(height)
                            .putArg(2*width)
                            .putArg(2*height)
                            .putArg(targetwidth)
                            .putArg(targetheight)
                            .putArg((float)((TranslationTransformation)scat.transformation).offsetx)
                            .putArg((float)((TranslationTransformation)scat.transformation).offsety);
                }
                else
                {
                    uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation]
                            .putArg(entryImageBuffer)
                            .putArg(fullSizedGPUResidentHelperBuffer)
                            .putArg(width)
                            .putArg(height)
                            .putArg(2*width)
                            .putArg(2*height)
                            .putArg(targetwidth)
                            .putArg(targetheight)
                            .putArg(((TranslationTransformation)scat.transformation).offsetx)
                            .putArg(((TranslationTransformation)scat.transformation).offsety);
                }
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation].rewind();
            }
            else
            {
                /*
                Now the B-spline coefficients are in the entryImageBuffer,
                lastly the image needs to be transformed, rescaled back and then 
                convert it back to the original image format.
                */
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_translationTransformImageWithBsplineInterpolation].getWorkGroupSize(device);  // Local work size dimensions
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_translationTransformImageWithBsplineInterpolation], width*height);   // rounded up to the nearest multiple of the localWorkSize
                if(usesFloatGPU)
                {
                    uniformBSplineTransformProgramKernels[KERNEL_translationTransformImageWithBsplineInterpolation]
                            .putArg(entryImageBuffer)
                            .putArg(fullSizedGPUResidentHelperBuffer)
                            .putArg(width)
                            .putArg(height)
                            .putArg(width * 2)
                            .putArg(height * 2)
                            .putArg((float)((TranslationTransformation)scat.transformation).offsetx)
                            .putArg((float)((TranslationTransformation)scat.transformation).offsety);
                }
                else
                {
                    uniformBSplineTransformProgramKernels[KERNEL_translationTransformImageWithBsplineInterpolation]
                            .putArg(entryImageBuffer)
                            .putArg(fullSizedGPUResidentHelperBuffer)
                            .putArg(width)
                            .putArg(height)
                            .putArg(width * 2)
                            .putArg(height * 2)
                            .putArg(((TranslationTransformation)scat.transformation).offsetx)
                            .putArg(((TranslationTransformation)scat.transformation).offsety);
                }
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_translationTransformImageWithBsplineInterpolation],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_translationTransformImageWithBsplineInterpolation].rewind();
            }
            // Transformed and rescaled back to an image at the same time now convert it back to the original format
            fetchTransformedImage();
        }
        
        private void transformImageRigidBody()
        {
            int width = (int)sharedContext.img.dimension(0);
            int height = (int)sharedContext.img.dimension(1);
            int localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].getWorkGroupSize(device);  // Local work size dimensions
            int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2Dpremulhp], width*height);   // rounded up to the nearest multiple of the localWorkSize
            // pre-multiply the image
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].putArg(entryImageBuffer).putArg(width*height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp],0,globalWorkSize,localWorkSize);
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp].getWorkGroupSize(device);
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp].putArg(entryImageBuffer).putArg(width).putArg(height);
            // Conversion to B-spline coefficients along X axis (Group size must be >= height)
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DXhp], height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp].rewind();
            // Now along the Y-axis
            // Has to be pre-multiplied by lambda again!!!
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2Dpremulhp], width*height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp].putArg(entryImageBuffer).putArg(width).putArg(height);
            // Conversion to B-spline coefficients along Y axis (Group size must be >= width)
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DYhp], width);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp].rewind();
            
            if(sharedContext.getResizeAfterRegistration())
            {
                int targetwidth = (int)sharedContext.resizedTargetImage.dimension(0);
                int targetheight = (int)sharedContext.resizedTargetImage.dimension(1);
                /*
                Now the B-spline coefficients are in the entryImageBuffer,
                lastly the image needs to be transformed, rescaled back and then 
                convert it back to the original image format.
                */
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation].getWorkGroupSize(device);  // Local work size dimensions
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_resizeTransformImageWithBsplineInterpolation], targetwidth*targetheight);   // rounded up to the nearest multiple of the localWorkSize
                if(usesFloatGPU)
                {
                    uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation]
                            .putArg(entryImageBuffer)
                            .putArg(fullSizedGPUResidentHelperBuffer)
                            .putArg(width)
                            .putArg(height)
                            .putArg(2*width)
                            .putArg(2*height)
                            .putArg(targetwidth)
                            .putArg(targetheight)
                            .putArg((float)((RigidBodyTransformation)scat.transformation).offsetx)
                            .putArg((float)((RigidBodyTransformation)scat.transformation).offsety)
                            .putArg((float)Math.cos(((RigidBodyTransformation)scat.transformation).angle))
                            .putArg((float)-Math.sin(((RigidBodyTransformation)scat.transformation).angle));
                }
                else
                {
                    uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation]
                            .putArg(entryImageBuffer)
                            .putArg(fullSizedGPUResidentHelperBuffer)
                            .putArg(width)
                            .putArg(height)
                            .putArg(2*width)
                            .putArg(2*height)
                            .putArg(targetwidth)
                            .putArg(targetheight)
                            .putArg(((RigidBodyTransformation)scat.transformation).offsetx)
                            .putArg(((RigidBodyTransformation)scat.transformation).offsety)
                            .putArg(Math.cos(((RigidBodyTransformation)scat.transformation).angle))
                            .putArg(-Math.sin(((RigidBodyTransformation)scat.transformation).angle));
                }
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation].rewind();
            }
            else
            {
                /*
                Now the B-spline coefficients are in the entryImageBuffer,
                lastly the image needs to be transformed, rescaled back and then 
                convert it back to the original image format.
                */
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_transformImageWithBsplineInterpolation].getWorkGroupSize(device);  // Local work size dimensions
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_transformImageWithBsplineInterpolation], width*height);   // rounded up to the nearest multiple of the localWorkSize
                if(usesFloatGPU)
                {
                    uniformBSplineTransformProgramKernels[KERNEL_transformImageWithBsplineInterpolation]
                            .putArg(entryImageBuffer)
                            .putArg(fullSizedGPUResidentHelperBuffer)
                            .putArg(width)
                            .putArg(height)
                            .putArg(2*width)
                            .putArg(2*height)
                            .putArg((float)((RigidBodyTransformation)scat.transformation).offsetx)
                            .putArg((float)((RigidBodyTransformation)scat.transformation).offsety)
                            .putArg((float)Math.cos(((RigidBodyTransformation)scat.transformation).angle))
                            .putArg((float)-Math.sin(((RigidBodyTransformation)scat.transformation).angle));
                }
                else
                {
                    uniformBSplineTransformProgramKernels[KERNEL_transformImageWithBsplineInterpolation]
                            .putArg(entryImageBuffer)
                            .putArg(fullSizedGPUResidentHelperBuffer)
                            .putArg(width)
                            .putArg(height)
                            .putArg(2*width)
                            .putArg(2*height)
                            .putArg(((RigidBodyTransformation)scat.transformation).offsetx)
                            .putArg(((RigidBodyTransformation)scat.transformation).offsety)
                            .putArg(Math.cos(((RigidBodyTransformation)scat.transformation).angle))
                            .putArg(-Math.sin(((RigidBodyTransformation)scat.transformation).angle));
                }
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_transformImageWithBsplineInterpolation],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_transformImageWithBsplineInterpolation].rewind();
            }
            
            // Transformed and rescaled back to an image at the same time now convert it back to the original format
            fetchTransformedImage();
        }
        
        private void transformImageScaledRotation()
        {
            int width = (int)sharedContext.img.dimension(0);
            int height = (int)sharedContext.img.dimension(1);
            int localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].getWorkGroupSize(device);  // Local work size dimensions
            int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2Dpremulhp], width*height);   // rounded up to the nearest multiple of the localWorkSize
            // pre-multiply the image
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].putArg(entryImageBuffer).putArg(width*height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp],0,globalWorkSize,localWorkSize);
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp].getWorkGroupSize(device);
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp].putArg(entryImageBuffer).putArg(width).putArg(height);
            // Conversion to B-spline coefficients along X axis (Group size must be >= height)
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DXhp], height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp].rewind();
            // Now along Y-axis
            // Has to be pre-multiplied by lambda again!!!
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2Dpremulhp], width*height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp].putArg(entryImageBuffer).putArg(width).putArg(height);
            // Conversion to B-spline coefficients along Y axis (Group size must be >= width)
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DYhp], width);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp].rewind();
            
            if(sharedContext.getResizeAfterRegistration())
            {
                int targetwidth = (int)sharedContext.resizedTargetImage.dimension(0);
                int targetheight = (int)sharedContext.resizedTargetImage.dimension(1);
                /*
                Now the B-spline coefficients are in the entryImageBuffer,
                lastly the image needs to be transformed, rescaled back and then 
                convert it back to the original image format.
                */
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation].getWorkGroupSize(device);  // Local work size dimensions
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_resizeTransformImageWithBsplineInterpolation], targetwidth*targetheight);   // rounded up to the nearest multiple of the localWorkSize
                if(usesFloatGPU)
                {
                    uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation]
                            .putArg(entryImageBuffer)
                            .putArg(fullSizedGPUResidentHelperBuffer)
                            .putArg(width)
                            .putArg(height)
                            .putArg(2*width)
                            .putArg(2*height)
                            .putArg(targetwidth)
                            .putArg(targetheight)
                            .putArg((float)((ScaledRotationTransformation)scat.transformation).offsetx)
                            .putArg((float)((ScaledRotationTransformation)scat.transformation).offsety)
                            .putArg((float)Math.cos(((ScaledRotationTransformation)scat.transformation).angle))
                            .putArg((float)-Math.sin(((ScaledRotationTransformation)scat.transformation).angle))
                            .putArg((float)((ScaledRotationTransformation)scat.transformation).scale);
                }
                else
                {
                    uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation]
                            .putArg(entryImageBuffer)
                            .putArg(fullSizedGPUResidentHelperBuffer)
                            .putArg(width)
                            .putArg(height)
                            .putArg(2*width)
                            .putArg(2*height)
                            .putArg(targetwidth)
                            .putArg(targetheight)
                            .putArg(((ScaledRotationTransformation)scat.transformation).offsetx)
                            .putArg(((ScaledRotationTransformation)scat.transformation).offsety)
                            .putArg(Math.cos(((ScaledRotationTransformation)scat.transformation).angle))
                            .putArg(-Math.sin(((ScaledRotationTransformation)scat.transformation).angle))
                            .putArg(((ScaledRotationTransformation)scat.transformation).scale);
                }
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation].rewind();
            }
            else
            {
                /*
                Now the B-spline coefficients are in the entryImageBuffer,
                lastly the image needs to be transformed, rescaled back and then 
                convert it back to the original image format.
                */
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_scaledRotationTransformImageWithBsplineInterpolation].getWorkGroupSize(device);  // Local work size dimensions
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_scaledRotationTransformImageWithBsplineInterpolation], width*height);   // rounded up to the nearest multiple of the localWorkSize
                if(usesFloatGPU)
                {
                    uniformBSplineTransformProgramKernels[KERNEL_scaledRotationTransformImageWithBsplineInterpolation]
                            .putArg(entryImageBuffer)
                            .putArg(fullSizedGPUResidentHelperBuffer)
                            .putArg(width)
                            .putArg(height)
                            .putArg(width * 2)
                            .putArg(height * 2)
                            .putArg((float)((ScaledRotationTransformation)scat.transformation).offsetx)
                            .putArg((float)((ScaledRotationTransformation)scat.transformation).offsety)
                            .putArg((float)Math.cos(((ScaledRotationTransformation)scat.transformation).angle))
                            .putArg((float)-Math.sin(((ScaledRotationTransformation)scat.transformation).angle))
                            .putArg((float)((ScaledRotationTransformation)scat.transformation).scale);
                }
                else
                {
                    uniformBSplineTransformProgramKernels[KERNEL_scaledRotationTransformImageWithBsplineInterpolation]
                            .putArg(entryImageBuffer)
                            .putArg(fullSizedGPUResidentHelperBuffer)
                            .putArg(width)
                            .putArg(height)
                            .putArg(width * 2)
                            .putArg(height * 2)
                            .putArg(((ScaledRotationTransformation)scat.transformation).offsetx)
                            .putArg(((ScaledRotationTransformation)scat.transformation).offsety)
                            .putArg(Math.cos(((ScaledRotationTransformation)scat.transformation).angle))
                            .putArg(-Math.sin(((ScaledRotationTransformation)scat.transformation).angle))
                            .putArg(((ScaledRotationTransformation)scat.transformation).scale);
                }
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_scaledRotationTransformImageWithBsplineInterpolation],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_scaledRotationTransformImageWithBsplineInterpolation].rewind();
            }
            // Transformed and rescaled back to an image at the same time now convert it back to the original format
            fetchTransformedImage();
        }
        
        private void transformImageAffine()
        {
            int width = (int)sharedContext.img.dimension(0);
            int height = (int)sharedContext.img.dimension(1);
            int localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].getWorkGroupSize(device);  // Local work size dimensions
            int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2Dpremulhp], width*height);   // rounded up to the nearest multiple of the localWorkSize
            // pre-multiply the image
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].putArg(entryImageBuffer).putArg(width*height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp],0,globalWorkSize,localWorkSize);
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp].getWorkGroupSize(device);
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp].putArg(entryImageBuffer).putArg(width).putArg(height);
            // Conversion to B-spline coefficients along X axis (Group size must be >= height)
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DXhp], height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp].rewind();
            // Now along Y-axis
            // Has to be pre-multiplied by lambda again!!!
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2Dpremulhp], width*height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp].putArg(entryImageBuffer).putArg(width).putArg(height);
            // Conversion to B-spline coefficients along Y axis (Group size must be >= width)
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DYhp], width);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp].rewind();
            
            if(sharedContext.getResizeAfterRegistration())
            {
                int targetwidth = (int)sharedContext.resizedTargetImage.dimension(0);
                int targetheight = (int)sharedContext.resizedTargetImage.dimension(1);
                /*
                Now the B-spline coefficients are in the entryImageBuffer,
                lastly the image needs to be transformed, rescaled back and then 
                convert it back to the original image format.
                */
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation].getWorkGroupSize(device);  // Local work size dimensions
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_resizeTransformImageWithBsplineInterpolation], targetwidth*targetheight);   // rounded up to the nearest multiple of the localWorkSize
                if(usesFloatGPU)
                {
                    uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation]
                            .putArg(entryImageBuffer)
                            .putArg(fullSizedGPUResidentHelperBuffer)
                            .putArg(width)
                            .putArg(height)
                            .putArg(2*width)
                            .putArg(2*height)
                            .putArg(targetwidth)
                            .putArg(targetheight)
                            .putArg((float)((AffineTransformation)scat.transformation).offsetx)
                            .putArg((float)((AffineTransformation)scat.transformation).offsety)
                            .putArg((float)((AffineTransformation)scat.transformation).a11)
                            .putArg((float)((AffineTransformation)scat.transformation).a12)
                            .putArg((float)((AffineTransformation)scat.transformation).a21)
                            .putArg((float)((AffineTransformation)scat.transformation).a22);
                }
                else
                {
                    uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation]
                            .putArg(entryImageBuffer)
                            .putArg(fullSizedGPUResidentHelperBuffer)
                            .putArg(width)
                            .putArg(height)
                            .putArg(2*width)
                            .putArg(2*height)
                            .putArg(targetwidth)
                            .putArg(targetheight)
                            .putArg(((AffineTransformation)scat.transformation).offsetx)
                            .putArg(((AffineTransformation)scat.transformation).offsety)
                            .putArg(((AffineTransformation)scat.transformation).a11)
                            .putArg(((AffineTransformation)scat.transformation).a12)
                            .putArg(((AffineTransformation)scat.transformation).a21)
                            .putArg(((AffineTransformation)scat.transformation).a22);
                }
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_resizeTransformImageWithBsplineInterpolation].rewind();
            }
            else
            {
                /*
                Now the B-spline coefficients are in the entryImageBuffer,
                lastly the image needs to be transformed, rescaled back and then 
                convert it back to the original image format.
                */
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_affineTransformImageWithBsplineInterpolation].getWorkGroupSize(device);  // Local work size dimensions
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_affineTransformImageWithBsplineInterpolation], width*height);   // rounded up to the nearest multiple of the localWorkSize
                if(usesFloatGPU)
                {
                    uniformBSplineTransformProgramKernels[KERNEL_affineTransformImageWithBsplineInterpolation]
                            .putArg(entryImageBuffer)
                            .putArg(fullSizedGPUResidentHelperBuffer)
                            .putArg(width)
                            .putArg(height)
                            .putArg(width * 2)
                            .putArg(height * 2)
                            .putArg((float)((AffineTransformation)scat.transformation).offsetx)
                            .putArg((float)((AffineTransformation)scat.transformation).offsety)
                            .putArg((float)((AffineTransformation)scat.transformation).a11)
                            .putArg((float)((AffineTransformation)scat.transformation).a12)
                            .putArg((float)((AffineTransformation)scat.transformation).a21)
                            .putArg((float)((AffineTransformation)scat.transformation).a22);
                }
                else
                {
                    uniformBSplineTransformProgramKernels[KERNEL_affineTransformImageWithBsplineInterpolation]
                            .putArg(entryImageBuffer)
                            .putArg(fullSizedGPUResidentHelperBuffer)
                            .putArg(width)
                            .putArg(height)
                            .putArg(width * 2)
                            .putArg(height * 2)
                            .putArg(((AffineTransformation)scat.transformation).offsetx)
                            .putArg(((AffineTransformation)scat.transformation).offsety)
                            .putArg(((AffineTransformation)scat.transformation).a11)
                            .putArg(((AffineTransformation)scat.transformation).a12)
                            .putArg(((AffineTransformation)scat.transformation).a21)
                            .putArg(((AffineTransformation)scat.transformation).a22);
                }
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_affineTransformImageWithBsplineInterpolation],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_affineTransformImageWithBsplineInterpolation].rewind();
            }
            // Transformed and rescaled back to an image at the same time now convert it back to the original format
            fetchTransformedImage();
        }
        
        private void fetchTransformedImage()
        {
            if(!sharedContext.getResizeAfterRegistration())
            {
                // WARNING: this function overwrites the original image data without a chance to recover it!!!
                if((sharedContext.img.firstElement() instanceof FloatType))
                {
                    if(usesFloatGPU)
                    {
                        ByteBuffer buffer = queue.putMapBuffer(fullSizedGPUResidentHelperBuffer, CLMemory.Map.READ, true);
                        buffer.rewind();
                        buffer.asFloatBuffer().get((float[]) scat.targetArray);
                        queue.putUnmapMemory(fullSizedGPUResidentHelperBuffer, buffer);
                        return;
                    }
                }
                if ((sharedContext.img.firstElement() instanceof DoubleType)) {
                    // Due to the pretest this makes usesFloatGPU = false
                    ByteBuffer buffer = queue.putMapBuffer(fullSizedGPUResidentHelperBuffer, CLMemory.Map.READ, true);
                    buffer.rewind();
                    buffer.asDoubleBuffer().get((double[]) scat.targetArray);
                    queue.putUnmapMemory(fullSizedGPUResidentHelperBuffer, buffer);
                    return;
                }
                
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
                } else if ((sharedContext.img.firstElement() instanceof LongType)||(sharedContext.img.firstElement() instanceof UnsignedLongType)) {
                    ByteBuffer buffer = queue.putMapBuffer(conversionEntryBuffer, CLMemory.Map.READ, true);
                    buffer.rewind();
                    buffer.asLongBuffer().get((long[]) scat.targetArray);
                    queue.putUnmapMemory(conversionEntryBuffer, buffer);
                } else if ((sharedContext.img.firstElement() instanceof FloatType)) {
                    ByteBuffer buffer = queue.putMapBuffer(conversionEntryBuffer, CLMemory.Map.READ, true);
                    buffer.rewind();
                    buffer.asFloatBuffer().get((float[]) scat.targetArray);
                    queue.putUnmapMemory(conversionEntryBuffer, buffer);
                }
            }
            else
            {
                //Resized target is in source not target
                if((sharedContext.resizedTargetImage.firstElement() instanceof FloatType))
                {
                    if(usesFloatGPU)
                    {
                        ByteBuffer buffer = queue.putMapBuffer(fullSizedGPUResidentHelperBuffer, CLMemory.Map.READ, true);
                        buffer.rewind();
                        buffer.asFloatBuffer().get((float[]) scat.sourceArray);
                        queue.putUnmapMemory(fullSizedGPUResidentHelperBuffer, buffer);
                        return;
                    }
                }
                if ((sharedContext.img.firstElement() instanceof DoubleType)) {
                    // Due to the pretest this makes usesFloatGPU = false
                    ByteBuffer buffer = queue.putMapBuffer(fullSizedGPUResidentHelperBuffer, CLMemory.Map.READ, true);
                    buffer.rewind();
                    buffer.asDoubleBuffer().get((double[]) scat.sourceArray);
                    queue.putUnmapMemory(fullSizedGPUResidentHelperBuffer, buffer);
                    return;
                }
                
                // The following converts the image to its original format and transfers it to conversionEntryBuffer
                int localWorkSize = (int)deConversionProgramKernel.getWorkGroupSize(device);  // Local work size dimensions
                int globalWorkSize = StaticUtility.roundUp(localWorkSize, deConversionProgramKernel.getPreferredWorkGroupSizeMultiple(device),(int) sharedContext.resizedTargetImage.dimension(0)*(int)sharedContext.resizedTargetImage.dimension(1));   // rounded up to the nearest multiple of the localWorkSize
                queue.put1DRangeKernel(deConversionProgramKernel,0,globalWorkSize,localWorkSize);
                if ((sharedContext.resizedTargetImage.firstElement() instanceof ByteType)||(sharedContext.resizedTargetImage.firstElement() instanceof UnsignedByteType)) {
                    ByteBuffer buffer = queue.putMapBuffer(conversionEntryBuffer, CLMemory.Map.READ, true);
                    buffer.rewind();
                    buffer.get((byte[]) scat.sourceArray);
                    queue.putUnmapMemory(conversionEntryBuffer, buffer);
                } else if ((sharedContext.resizedTargetImage.firstElement() instanceof ShortType)||(sharedContext.resizedTargetImage.firstElement() instanceof UnsignedShortType)) {
                    ByteBuffer buffer = queue.putMapBuffer(conversionEntryBuffer, CLMemory.Map.READ, true);
                    buffer.rewind();
                    buffer.asShortBuffer().get((short[]) scat.sourceArray);
                    queue.putUnmapMemory(conversionEntryBuffer, buffer);
                } else if ((sharedContext.resizedTargetImage.firstElement() instanceof IntType)||(sharedContext.resizedTargetImage.firstElement() instanceof UnsignedIntType)) {
                    ByteBuffer buffer = queue.putMapBuffer(conversionEntryBuffer, CLMemory.Map.READ, true);
                    buffer.rewind();
                    buffer.asIntBuffer().get((int[]) scat.sourceArray);
                    queue.putUnmapMemory(conversionEntryBuffer, buffer);
                } else if ((sharedContext.resizedTargetImage.firstElement() instanceof LongType)||(sharedContext.resizedTargetImage.firstElement() instanceof UnsignedLongType)) {
                    ByteBuffer buffer = queue.putMapBuffer(conversionEntryBuffer, CLMemory.Map.READ, true);
                    buffer.rewind();
                    buffer.asLongBuffer().get((long[]) scat.sourceArray);
                    queue.putUnmapMemory(conversionEntryBuffer, buffer);
                } else if ((sharedContext.resizedTargetImage.firstElement() instanceof FloatType)) {
                    ByteBuffer buffer = queue.putMapBuffer(conversionEntryBuffer, CLMemory.Map.READ, true);
                    buffer.rewind();
                    buffer.asFloatBuffer().get((float[]) scat.sourceArray);
                    queue.putUnmapMemory(conversionEntryBuffer, buffer);
                }
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
                break;
            case RIGIDBODY:
                for(int i = pyramidDepth - 1;i > 0;i--)
                {
                    iterationPower /= 2;
                    inverseMarquardtLevenbergRigidBodyOptimization(i);
                    // scale up (but the rotation is not scale dependent)
                    offsetx *= 2.0;
                    offsety *= 2.0;
                }
                iterationPower /= 2;
                inverseMarquardtLevenbergRigidBodyOptimization(0);
                break;
            case SCALEDROTATION:
                for(int i = pyramidDepth - 1;i > 0;i--)
                {
                    iterationPower /= 2;
                    inverseMarquardtLevenbergScaledRotationOptimization(i);
                    // scale up (but the rotation and scaling is not scale dependent)
                    offsetx *= 2.0;
                    offsety *= 2.0;
                }
                iterationPower /= 2;
                inverseMarquardtLevenbergScaledRotationOptimization(0);
                break;
            case AFFINE:
                for(int i = pyramidDepth - 1;i > 0;i--)
                {
                    iterationPower /= 2;
                    inverseMarquardtLevenbergAffineOptimization(i);
                    // scale up (but the affine parameters are not scale dependent)
                    offsetx *= 2.0;
                    offsety *= 2.0;
                }
                iterationPower /= 2;
                inverseMarquardtLevenbergAffineOptimization(0);
                break;
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
            // first initialize the matrix with the current transformation (upscaling between the steps)
            double currentoffsetx;
            double currentoffsety;
            double currentangle;
            bestMeanSquares = getTranslationMeanSquares(pyramidIndex,offsetx,offsety);
            iteration++;
            do {
                // calculate the pseudo hessian from the hessian
                for (int k = 0; (k < 2); k++) {
                    Arrays.fill(pseudoHessian[k], 0.0);
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
                    Arrays.fill(pseudoHessian[k], 0.0);
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

        private void inverseMarquardtLevenbergScaledRotationOptimization(int pyramidIndex)
        {
            double[] update = {0.0,0.0,0.0,0.0};
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
            double currentscale;
            bestMeanSquares = getScaledRotationMeanSquares(pyramidIndex,offsetx,offsety,this.angle,this.scale);
            iteration++;
            do {
                // calculate the pseudo hessian from the hessian
                for (int k = 0; (k < 4); k++) {
                    Arrays.fill(pseudoHessian[k], 0.0);
                    pseudoHessian[k][k] = (1.0 + lambda) * hessian[k][k];
                }
                StaticUtility.invertGauss(pseudoHessian);
                update = StaticUtility.matrixMultiply(pseudoHessian, gradient);
                currentscale = this.scale + update[0];
                currentangle = this.angle - update[1];
                displacement = Math.sqrt(update[2] * update[2] + update[3] * update[3]) + 0.25 * Math.sqrt((double)(targetPyramid[pyramidIndex].width * targetPyramid[pyramidIndex].width) + (double)(targetPyramid[pyramidIndex].height * targetPyramid[pyramidIndex].height)) * (Math.abs(update[0]) + Math.abs(update[1]));
                c = Math.cos(update[1]);
                s = Math.sin(update[1]);
                currentoffsetx = ((offsetx + update[2]) * c - (offsety + update[3]) * s) * (1.0 + update[0]);
                currentoffsety = ((offsetx + update[2]) * s + (offsety + update[3]) * c) * (1.0 + update[0]);
                meanSquares = getScaledRotationMeanSquares(pyramidIndex,currentoffsetx,currentoffsety,currentangle,currentscale);

                iteration++;
                if (meanSquares < bestMeanSquares) {
                    bestMeanSquares = meanSquares;
                    lambda /= 4.0;
                    offsetx = currentoffsetx;
                    offsety = currentoffsety;
                    this.angle = currentangle;
                    this.scale = currentscale;
                }
                else {
                    lambda *= 4.0;
                }
            } while ((iteration < (10 * iterationPower - 1)) && (0.001 <= displacement));
            StaticUtility.invertGauss(hessian);
            update = StaticUtility.matrixMultiply(hessian, gradient);
            currentscale = this.scale + update[0];
            currentangle = this.angle - update[1];
            c = Math.cos(update[1]);
            s = Math.sin(update[1]);
            currentoffsetx = ((offsetx + update[2]) * c - (offsety + update[3]) * s) * (1.0 + update[0]);
            currentoffsety = ((offsetx + update[2]) * s + (offsety + update[3]) * c) * (1.0 + update[0]);
            meanSquares = getScaledRotationMeanSquaresWithoutHessian(pyramidIndex,currentoffsetx,currentoffsety,currentangle,currentscale);
            iteration++;
            if (meanSquares < bestMeanSquares) {
                offsetx = currentoffsetx;
                offsety = currentoffsety;
                this.angle = currentangle;
                this.scale = currentscale;
            }
        }
        
        private void inverseMarquardtLevenbergAffineOptimization(int pyramidIndex)
        {
            double[] update = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            double bestMeanSquares = 0.0;
            double meanSquares = 0.0;
            double lambda = 1.0;
            double displacement;
            int iteration = 0;

            // Trial parameters for each LM step
            double currenta11;
            double currenta12;
            double currenta21;
            double currenta22;
            double currentoffsetx;
            double currentoffsety;
            
            /*
             * Initial evaluation: compute E(p₀), gradient g, and Hessian H
             * at the current parameter values. The gradient and Hessian are
             * stored in the class fields gradient[] and hessian[][].
             */
            bestMeanSquares = getAffineMeanSquares(pyramidIndex, offsetx, offsety, a11, a12, a21, a22);
            iteration++;
            do {
                /*
                 * Construct the LM-damped pseudoHessian.
                 * Only the diagonal is set: pseudoH[k][k] = (1+λ) * H[k][k].
                 * Off-diagonals remain zero (the pseudoHessian arrays are reused
                 * and the off-diagonals were never set to non-zero values).
                 * Inverting this diagonal matrix and multiplying by the gradient
                 * yields: δp_k = g_k / ((1+λ) * H_kk).
                 */
                for (int k = 0; (k < 6); k++) {
                    Arrays.fill(pseudoHessian[k], 0.0);
                    pseudoHessian[k][k] = (1.0 + lambda) * hessian[k][k];
                }
                StaticUtility.invertGauss(pseudoHessian);
                update = StaticUtility.matrixMultiply(pseudoHessian, gradient);
                
                /*
                 * Compute the displacement for convergence testing.
                 * The translational displacement is the Euclidean norm of (δtx, δty).
                 * The matrix-element displacement is heuristically converted to pixels
                 * by multiplying by 0.25 × image diagonal, analogous to the rigid body
                 * angle-to-pixel conversion.
                 */
                /*double diag = Math.sqrt((double)(targetPyramid[pyramidIndex].width * targetPyramid[pyramidIndex].width)
                                      + (double)(targetPyramid[pyramidIndex].height * targetPyramid[pyramidIndex].height));
                displacement = Math.sqrt(update[4] * update[4] + update[5] * update[5])
                             + 0.25 * diag * (Math.abs(update[0]) + Math.abs(update[1]) + Math.abs(update[2]) + Math.abs(update[3]));*/
                
                /*
                 * A change δa11 causes a displacement of approximately |δa11| * |x| at pixel position x. 
                 * The maximum |x| is approximately width/2 (measuring from center). 
                 * Similarly, δa12 causes |δa12| * |y| where max |y| ≈ height/2. 
                 * So a tighter heuristic would be:
                 */
                displacement = Math.sqrt(update[4] * update[4] + update[5] * update[5])
                        + 0.5 * (double)targetPyramid[pyramidIndex].width
                          * (Math.abs(update[0]) + Math.abs(update[2]))
                        + 0.5 * (double)targetPyramid[pyramidIndex].height
                          * (Math.abs(update[1]) + Math.abs(update[3]));
                
                
                /*
                 * Affine parameter update: purely additive, positive sign.
                 * Unlike rigid body where angle is SUBTRACTED and offsets are COMPOSED
                 * through a rotation matrix, affine parameters form a vector space
                 * and are simply incremented.
                 */
                currenta11    = this.a11    + update[0];
                currenta12    = this.a12    + update[1];
                currenta21    = this.a21    + update[2];
                currenta22    = this.a22    + update[3];
                currentoffsetx = this.offsetx + update[4];
                currentoffsety = this.offsety + update[5];
                
                meanSquares = getAffineMeanSquares(pyramidIndex, currentoffsetx, currentoffsety,
                        currenta11, currenta12, currenta21, currenta22);

                iteration++;
                if (meanSquares < bestMeanSquares) {
                    /*
                     * Accept the step: the MSE decreased.
                     * Reduce λ (shift toward Gauss-Newton / larger steps).
                     */
                    bestMeanSquares = meanSquares;
                    lambda /= 4.0;
                    this.a11    = currenta11;
                    this.a12    = currenta12;
                    this.a21    = currenta21;
                    this.a22    = currenta22;
                    this.offsetx = currentoffsetx;
                    this.offsety = currentoffsety;
                }
                else {
                    /*
                     * Reject the step: the MSE did not decrease.
                     * Increase λ (shift toward gradient descent / smaller steps).
                     * The gradient and Hessian from the PREVIOUS accepted point
                     * are still valid (they were overwritten by getAffineMeanSquares
                     * but will be recomputed at the accepted point on the next
                     * iteration if the step is accepted).
                     */
                    lambda *= 4.0;
                }
            } while ((iteration < (10 * iterationPower - 1)) && (0.001 <= displacement));
            /*
             * Final undamped Gauss-Newton step: invert the FULL Hessian (with
             * off-diagonals) and apply one pure quadratic step. This gives the
             * best local quadratic approximation without LM damping.
             * Accept only if it improves the MSE.
             */
            StaticUtility.invertGauss(hessian);
            update = StaticUtility.matrixMultiply(hessian, gradient);

            currenta11     = this.a11     + update[0];
            currenta12     = this.a12     + update[1];
            currenta21     = this.a21     + update[2];
            currenta22     = this.a22     + update[3];
            currentoffsetx = this.offsetx + update[4];
            currentoffsety = this.offsety + update[5];

            meanSquares = getAffineMeanSquaresWithoutHessian(pyramidIndex, currentoffsetx, currentoffsety,
                                                              currenta11, currenta12, currenta21, currenta22);
            iteration++;
            if (meanSquares < bestMeanSquares) {
                this.a11     = currenta11;
                this.a12     = currenta12;
                this.a21     = currenta21;
                this.a22     = currenta22;
                this.offsetx = currentoffsetx;
                this.offsety = currentoffsety;
            }
        }
        
        private double getTranslationMeanSquares(int pyramidIndex, double currentoffsetx, double curentoffsety)
        {
            // First reset the global values which will not be reset in the loop
            Arrays.fill(gradient, 0.0);
            for (int i = 0; (i < 2); i++) {
                Arrays.fill(hessian[i], 0.0);
            }
            
            // now we have the diffs let's calculate all the derived elements
            int halfReductionSize = (int) (((sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)+((sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)%2))/2);
            ByteBuffer buffer;
            double area;
            // calculate the MSE row-wise
            double mse;
            if(halfReductionSize <= maximumElementsForLocalFPTcombinedSum)
            {                
                uniformBSplineTransformProgramKernels[KERNEL_translationErrorWithGradAndHess]
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
                        .putArg((int)targetPyramid[pyramidIndex].width*2)
                        .putArg((int)targetPyramid[pyramidIndex].height*2);
                
                if(usesFloatGPU)
                {
                    uniformBSplineTransformProgramKernels[KERNEL_translationErrorWithGradAndHess]
                            .putArg((float)currentoffsetx)
                            .putArg((float)curentoffsety);
                }
                else
                {
                    uniformBSplineTransformProgramKernels[KERNEL_translationErrorWithGradAndHess]
                            .putArg(currentoffsetx)
                            .putArg(curentoffsety);
                }
                int localWorkSize = (int)Math.min(uniformBSplineTransformProgramKernels[KERNEL_translationErrorWithGradAndHess].getWorkGroupSize(device),blocksizeMultiplier*optimalMultiples[KERNEL_translationErrorWithGradAndHess]);  // Local work size dimensions
                int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_translationErrorWithGradAndHess], (int)(sourcePyramid[pyramidIndex].width*sourcePyramid[pyramidIndex].height));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_translationErrorWithGradAndHess],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_translationErrorWithGradAndHess].rewind();
                
                /*
                Because each of the following kernels is synchronized by barriers on the GPU they can only be executed on a single compute device.
                Therefore using an async queue may increase the speed by allowing multiple kernels to execute in parallel (if local memory permits)
                */
                localWorkSize = halfReductionSize % optimalMultiples[KERNEL_translationSumInLocalMemoryCombined] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_translationSumInLocalMemoryCombined] - (halfReductionSize % optimalMultiples[KERNEL_translationSumInLocalMemoryCombined])), maximumElementsForLocalFPTcombinedSum);
                globalWorkSize = localWorkSize;
                uniformBSplineTransformProgramKernels[KERNEL_translationSumInLocalMemoryCombined]
                        .putArg(maskBuffer)
                        .putArg(entryImageBuffer)
                        .putArg(gradient0)
                        .putArg(gradient1)
                        .putArg(hessian00)
                        .putArg(hessian01)
                        .putArg(hessian11)
                        .putNullArg(localWorkSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_translationSumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_translationSumInLocalMemoryCombined].rewind();
                
                buffer = queue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                area = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(maskBuffer, buffer);
                buffer = queue.putMapBuffer(entryImageBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                mse = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(entryImageBuffer, buffer);
                buffer = queue.putMapBuffer(gradient0, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[0] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient0, buffer);
                buffer = queue.putMapBuffer(gradient1, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient1, buffer);
                buffer = queue.putMapBuffer(hessian00, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][0] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian00, buffer);
                buffer = queue.putMapBuffer(hessian01, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian01, buffer);
                buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[1][1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian11, buffer);
            }
            else
            {
                
                int nrOfBlocks = ((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))%blockSizesParallel;
                nrOfBlocks = nrOfBlocks == 0 ? (int)Math.min(((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))/blockSizesParallel,maximumSumReductionBlockNr) : (int)Math.min((((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))+(blockSizesParallel-nrOfBlocks))/blockSizesParallel,maximumSumReductionBlockNr);
                
                int globalWorkSize = nrOfBlocks * blockSizesParallel;
                uniformBSplineTransformProgramKernels[KERNEL_translationErrorWithGradAndHessBrent]
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
                
                
                uniformBSplineTransformProgramKernels[KERNEL_translationErrorWithGradAndHessBrent]
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/);
                
                
                uniformBSplineTransformProgramKernels[KERNEL_translationErrorWithGradAndHessBrent]
                        .putArg((int)sourcePyramid[pyramidIndex].width)
                        .putArg((int)sourcePyramid[pyramidIndex].height)
                        .putArg((int)targetPyramid[pyramidIndex].width)
                        .putArg((int)targetPyramid[pyramidIndex].height);
                
                if(usesFloatGPU)
                {
                    uniformBSplineTransformProgramKernels[KERNEL_translationErrorWithGradAndHessBrent]
                            .putArg((float)currentoffsetx)
                            .putArg((float)curentoffsety);
                }
                else
                {
                    uniformBSplineTransformProgramKernels[KERNEL_translationErrorWithGradAndHessBrent]
                            .putArg(currentoffsetx)
                            .putArg(curentoffsety);
                }                
                uniformBSplineTransformProgramKernels[KERNEL_translationErrorWithGradAndHessBrent]
                        .putArg((int)targetPyramid[pyramidIndex].width*2)
                        .putArg((int)targetPyramid[pyramidIndex].height*2);
                
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_translationErrorWithGradAndHessBrent],0,globalWorkSize,blockSizesParallel);
                uniformBSplineTransformProgramKernels[KERNEL_translationErrorWithGradAndHessBrent].rewind();
                
                // use the in-memory summations (also async) each with its own buffer to do the final reduction)
                halfReductionSize = (int) ((nrOfBlocks+(nrOfBlocks%2))/2);
                int localWorkSize = halfReductionSize % optimalMultiples[KERNEL_translationSumInLocalMemoryCombined] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_translationSumInLocalMemoryCombined] - (halfReductionSize % optimalMultiples[KERNEL_translationSumInLocalMemoryCombined])), maximumElementsForLocalFPTcombinedSum);
                globalWorkSize = localWorkSize;
                uniformBSplineTransformProgramKernels[KERNEL_translationSumInLocalMemoryCombined]
                        .putArg(gradient0)
                        .putArg(gradient1)
                        .putArg(hessian00)
                        .putArg(hessian01)
                        .putArg(hessian11)
                        .putArg(entryImageBuffer)
                        .putArg(maskBuffer)
                        .putNullArg(halfReductionSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putArg((int)(nrOfBlocks));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_translationSumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_translationSumInLocalMemoryCombined].rewind();
                
                buffer = queue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                area = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(maskBuffer, buffer);
                
                buffer = queue.putMapBuffer(entryImageBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                mse = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(entryImageBuffer, buffer);
                        
                buffer = queue.putMapBuffer(gradient0, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[0] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient0, buffer);
                        
                buffer = queue.putMapBuffer(gradient1, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient1, buffer);
                        
                buffer = queue.putMapBuffer(hessian00, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][0] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian00, buffer);
                        
                buffer = queue.putMapBuffer(hessian01, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian01, buffer);
                        
                buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[1][1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian11, buffer);
            }
            // symmetrize hessian
            for (int i = 1; (i < 2); i++) {
                for (int j = 0; (j < i); j++) {
                        hessian[i][j] = hessian[j][i];
                }
            }
            return mse/area;
        }
        
        private double getRigidBodyMeanSquares(int pyramidIndex, double currentoffsetx, double curentoffsety, double currentangle)
        {
            // First reset the global values which will not be reset in the loop
            Arrays.fill(gradient, 0.0);
            for (int i = 0; (i < 3); i++) {
                Arrays.fill(hessian[i], 0.0);
            }
            
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
                        .putArg((int)targetPyramid[pyramidIndex].width * 2)
                        .putArg((int)targetPyramid[pyramidIndex].height * 2);
                if(usesFloatGPU)
                {
                    uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHess]
                            .putArg((float)currentoffsetx)
                            .putArg((float)curentoffsety)
                            .putArg((float)Math.cos(currentangle))
                            .putArg((float)-Math.sin(currentangle));
                }
                else
                {
                    uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHess]
                            .putArg(currentoffsetx)
                            .putArg(curentoffsety)
                            .putArg(Math.cos(currentangle))
                            .putArg(-Math.sin(currentangle));
                }
                int localWorkSize = (int)Math.min(uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHess].getWorkGroupSize(device),blocksizeMultiplier*optimalMultiples[KERNEL_rigidBodyErrorWithGradAndHess]);  // Local work size dimensions
                int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_rigidBodyErrorWithGradAndHess], (int)(sourcePyramid[pyramidIndex].width*sourcePyramid[pyramidIndex].height));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHess],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHess].rewind();
                
                /*
                Because each of the following kernels is synchronized by barriers on the GPU they can only be executed on a single compute device.
                Therefore using an async queue may increase the speed by allowing multiple kernels to execute in parallel (if local memory permits)
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
                        .putNullArg(localWorkSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemoryCombined].rewind();
                
                buffer = queue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                area = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(maskBuffer, buffer);
                buffer = queue.putMapBuffer(entryImageBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                mse = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(entryImageBuffer, buffer);
                buffer = queue.putMapBuffer(gradient0, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[0] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient0, buffer);
                buffer = queue.putMapBuffer(gradient1, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient1, buffer);
                buffer = queue.putMapBuffer(gradient2, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient2, buffer);
                buffer = queue.putMapBuffer(hessian00, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][0] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian00, buffer);
                buffer = queue.putMapBuffer(hessian01, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian01, buffer);
                buffer = queue.putMapBuffer(hessian02, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian02, buffer);
                buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[1][1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian11, buffer);
                buffer = queue.putMapBuffer(hessian12, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[1][2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian12, buffer);
                buffer = queue.putMapBuffer(hessian22, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[2][2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian22, buffer);
            }
            else
            {
                
                int nrOfBlocks = ((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))%blockSizesParallel;
                nrOfBlocks = nrOfBlocks == 0 ? (int)Math.min(((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))/blockSizesParallel,maximumSumReductionBlockNr) : (int)Math.min((((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))+(blockSizesParallel-nrOfBlocks))/blockSizesParallel,maximumSumReductionBlockNr);
                
                int globalWorkSize = nrOfBlocks * blockSizesParallel;
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
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/);
                
                
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent]
                        .putArg((int)sourcePyramid[pyramidIndex].width)
                        .putArg((int)sourcePyramid[pyramidIndex].height)
                        .putArg((int)targetPyramid[pyramidIndex].width)
                        .putArg((int)targetPyramid[pyramidIndex].height);
                
                if(usesFloatGPU)
                {
                    uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent]
                            .putArg((float)currentoffsetx)
                            .putArg((float)curentoffsety)
                            .putArg((float)Math.cos(currentangle))
                            .putArg((float)-Math.sin(currentangle));
                }
                else
                {
                    uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent]
                            .putArg(currentoffsetx)
                            .putArg(curentoffsety)
                            .putArg(Math.cos(currentangle))
                            .putArg(-Math.sin(currentangle));
                }                
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent]
                        .putArg((int)targetPyramid[pyramidIndex].width*2)
                        .putArg((int)targetPyramid[pyramidIndex].height*2);
                
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent],0,globalWorkSize,blockSizesParallel);
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent].rewind();
                
                // use the in-memory summations (also async) each with its own buffer to do the final reduction)
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
                        .putNullArg(halfReductionSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putArg((int)(nrOfBlocks));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemoryCombined].rewind();
                
                buffer = queue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                area = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(maskBuffer, buffer);
                
                buffer = queue.putMapBuffer(entryImageBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                mse = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(entryImageBuffer, buffer);
                        
                buffer = queue.putMapBuffer(gradient0, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[0] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient0, buffer);
                        
                buffer = queue.putMapBuffer(gradient1, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient1, buffer);
                        
                buffer = queue.putMapBuffer(gradient2, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient2, buffer);
                        
                buffer = queue.putMapBuffer(hessian00, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][0] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian00, buffer);
                        
                buffer = queue.putMapBuffer(hessian01, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian01, buffer);
                        
                buffer = queue.putMapBuffer(hessian02, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian02, buffer);
                        
                buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[1][1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian11, buffer);
                        
                buffer = queue.putMapBuffer(hessian12, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[1][2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian12, buffer);
                        
                buffer = queue.putMapBuffer(hessian22, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[2][2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian22, buffer);
            }
            // symmetrize hessian
            for (int i = 1; (i < 3); i++) {
                for (int j = 0; (j < i); j++) {
                        hessian[i][j] = hessian[j][i];
                }
            }
            return mse/area;
        }
        
        private double getScaledRotationMeanSquares(int pyramidIndex, double currentoffsetx, double curentoffsety, double currentangle, double currentscale)
        {
            // First reset the global values which will not be reset in the loop
            Arrays.fill(gradient, 0.0);
            for (int i = 0; (i < 4); i++) {
                Arrays.fill(hessian[i], 0.0);
            }
            
            // now we have the diffs let's calculate all the derived elements
            int halfReductionSize = (int) (((sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)+((sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)%2))/2);
            ByteBuffer buffer;
            double area;
            // calculate the MSE row-wise
            double mse;
            if(halfReductionSize <= maximumElementsForLocalFPTcombinedSum)
            {                
                uniformBSplineTransformProgramKernels[KERNEL_scaledRotationErrorWithGradAndHess]
                        .putArg(sourcePyramid[pyramidIndex].Image)
                        .putArg(targetPyramid[pyramidIndex].Coefficient)
                        .putArg(sourcePyramid[pyramidIndex].xGradient)
                        .putArg(sourcePyramid[pyramidIndex].yGradient)
                        .putArg(gradient0)
                        .putArg(gradient1)
                        .putArg(gradient2)
                        .putArg(gradient3)
                        .putArg(hessian00)
                        .putArg(hessian01)
                        .putArg(hessian02)
                        .putArg(hessian03)
                        .putArg(hessian11)
                        .putArg(hessian12)
                        .putArg(hessian13)
                        .putArg(hessian22)
                        .putArg(hessian23)
                        .putArg(hessian33)
                        .putArg(entryImageBuffer)
                        .putArg(maskBuffer)
                        .putArg((int)sourcePyramid[pyramidIndex].width)
                        .putArg((int)sourcePyramid[pyramidIndex].height)
                        .putArg((int)targetPyramid[pyramidIndex].width)
                        .putArg((int)targetPyramid[pyramidIndex].height)
                        .putArg((int)targetPyramid[pyramidIndex].width * 2)
                        .putArg((int)targetPyramid[pyramidIndex].height * 2);
                if(usesFloatGPU)
                {
                    uniformBSplineTransformProgramKernels[KERNEL_scaledRotationErrorWithGradAndHess]
                        .putArg((float)currentoffsetx)
                        .putArg((float)curentoffsety)
                        .putArg((float)Math.cos(currentangle))
                        .putArg((float)-Math.sin(currentangle))
                        .putArg((float)currentscale);
                }
                else
                {
                    uniformBSplineTransformProgramKernels[KERNEL_scaledRotationErrorWithGradAndHess]
                        .putArg(currentoffsetx)
                        .putArg(curentoffsety)
                        .putArg(Math.cos(currentangle))
                        .putArg(-Math.sin(currentangle))
                        .putArg(currentscale);
                }
                int localWorkSize = (int)Math.min(uniformBSplineTransformProgramKernels[KERNEL_scaledRotationErrorWithGradAndHess].getWorkGroupSize(device),blocksizeMultiplier*optimalMultiples[KERNEL_scaledRotationErrorWithGradAndHess]);  // Local work size dimensions
                int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_scaledRotationErrorWithGradAndHess], (int)(sourcePyramid[pyramidIndex].width*sourcePyramid[pyramidIndex].height));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_scaledRotationErrorWithGradAndHess],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_scaledRotationErrorWithGradAndHess].rewind();
                
                /*
                Because each of the following kernels is synchronized by barriers on the GPU they can only be executed on a single compute device.
                Therefore using an async queue may increase the speed by allowing multiple kernels to execute in parallel (if local memory permits)
                */
                localWorkSize = halfReductionSize % optimalMultiples[KERNEL_scaledRotationSumInLocalMemoryCombined] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_scaledRotationSumInLocalMemoryCombined] - (halfReductionSize % optimalMultiples[KERNEL_scaledRotationSumInLocalMemoryCombined])), maximumElementsForLocalFPTcombinedSum);
                globalWorkSize = localWorkSize;
                uniformBSplineTransformProgramKernels[KERNEL_scaledRotationSumInLocalMemoryCombined]
                        .putArg(maskBuffer)
                        .putArg(entryImageBuffer)
                        .putArg(gradient0)
                        .putArg(gradient1)
                        .putArg(gradient2)
                        .putArg(gradient3)
                        .putArg(hessian00)
                        .putArg(hessian01)
                        .putArg(hessian02)
                        .putArg(hessian03)
                        .putArg(hessian11)
                        .putArg(hessian12)
                        .putArg(hessian13)
                        .putArg(hessian22)
                        .putArg(hessian23)
                        .putArg(hessian33)
                        .putNullArg(localWorkSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_scaledRotationSumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_scaledRotationSumInLocalMemoryCombined].rewind();
                
                //TODO: the implicit if's for float/double can be reduced by separating the cases into two blocks using a single if statement
                buffer = queue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                area = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(maskBuffer, buffer);
                buffer = queue.putMapBuffer(entryImageBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                mse = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(entryImageBuffer, buffer);
                buffer = queue.putMapBuffer(gradient0, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[0] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient0, buffer);
                buffer = queue.putMapBuffer(gradient1, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient1, buffer);
                buffer = queue.putMapBuffer(gradient2, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient2, buffer);
                buffer = queue.putMapBuffer(gradient3, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[3] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient3, buffer);
                buffer = queue.putMapBuffer(hessian00, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][0] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian00, buffer);
                buffer = queue.putMapBuffer(hessian01, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian01, buffer);
                buffer = queue.putMapBuffer(hessian02, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian02, buffer);
                buffer = queue.putMapBuffer(hessian03, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][3] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian03, buffer);
                buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[1][1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian11, buffer);
                buffer = queue.putMapBuffer(hessian12, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[1][2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian12, buffer);
                buffer = queue.putMapBuffer(hessian13, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[1][3] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian13, buffer);
                buffer = queue.putMapBuffer(hessian22, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[2][2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian22, buffer);
                buffer = queue.putMapBuffer(hessian23, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[2][3] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian23, buffer);
                buffer = queue.putMapBuffer(hessian33, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[3][3] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian33, buffer);
            }
            else
            {
                
                int nrOfBlocks = ((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))%blockSizesParallel;
                nrOfBlocks = nrOfBlocks == 0 ? (int)Math.min(((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))/blockSizesParallel,maximumSumReductionBlockNr) : (int)Math.min((((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))+(blockSizesParallel-nrOfBlocks))/blockSizesParallel,maximumSumReductionBlockNr);
                
                int globalWorkSize = nrOfBlocks * blockSizesParallel;
                uniformBSplineTransformProgramKernels[KERNEL_scaledRotationErrorWithGradAndHessBrent]
                        .putArg(sourcePyramid[pyramidIndex].Image)
                        .putArg(targetPyramid[pyramidIndex].Coefficient)
                        .putArg(sourcePyramid[pyramidIndex].xGradient)
                        .putArg(sourcePyramid[pyramidIndex].yGradient)
                        .putArg(gradient0)
                        .putArg(gradient1)
                        .putArg(gradient2)
                        .putArg(gradient3)
                        .putArg(hessian00)
                        .putArg(hessian01)
                        .putArg(hessian02)
                        .putArg(hessian03)
                        .putArg(hessian11)
                        .putArg(hessian12)
                        .putArg(hessian13)
                        .putArg(hessian22)
                        .putArg(hessian23)
                        .putArg(hessian33)
                        .putArg(entryImageBuffer)
                        .putArg(maskBuffer);
                
                
                uniformBSplineTransformProgramKernels[KERNEL_scaledRotationErrorWithGradAndHessBrent]
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/);
                
                
                uniformBSplineTransformProgramKernels[KERNEL_scaledRotationErrorWithGradAndHessBrent]
                        .putArg((int)sourcePyramid[pyramidIndex].width)
                        .putArg((int)sourcePyramid[pyramidIndex].height)
                        .putArg((int)targetPyramid[pyramidIndex].width)
                        .putArg((int)targetPyramid[pyramidIndex].height);
                
                if(usesFloatGPU)
                {
                    uniformBSplineTransformProgramKernels[KERNEL_scaledRotationErrorWithGradAndHessBrent]
                        .putArg((float)currentoffsetx)
                        .putArg((float)curentoffsety)
                        .putArg((float)Math.cos(currentangle))
                        .putArg((float)-Math.sin(currentangle))
                        .putArg((float)currentscale);
                }
                else
                {
                    uniformBSplineTransformProgramKernels[KERNEL_scaledRotationErrorWithGradAndHessBrent]
                        .putArg(currentoffsetx)
                        .putArg(curentoffsety)
                        .putArg(Math.cos(currentangle))
                        .putArg(-Math.sin(currentangle))
                        .putArg(currentscale);
                }                
                uniformBSplineTransformProgramKernels[KERNEL_scaledRotationErrorWithGradAndHessBrent]
                        .putArg((int)targetPyramid[pyramidIndex].width*2)
                        .putArg((int)targetPyramid[pyramidIndex].height*2);
                
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_scaledRotationErrorWithGradAndHessBrent],0,globalWorkSize,blockSizesParallel);
                uniformBSplineTransformProgramKernels[KERNEL_scaledRotationErrorWithGradAndHessBrent].rewind();
                
                // use the in-memory summations (also async) each with its own buffer to do the final reduction)
                halfReductionSize = (int) ((nrOfBlocks+(nrOfBlocks%2))/2);
                int localWorkSize = halfReductionSize % optimalMultiples[KERNEL_scaledRotationSumInLocalMemoryCombined] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_scaledRotationSumInLocalMemoryCombined] - (halfReductionSize % optimalMultiples[KERNEL_scaledRotationSumInLocalMemoryCombined])), maximumElementsForLocalFPTcombinedSum);
                globalWorkSize = localWorkSize;
                uniformBSplineTransformProgramKernels[KERNEL_scaledRotationSumInLocalMemoryCombined]
                        .putArg(gradient0)
                        .putArg(gradient1)
                        .putArg(gradient2)
                        .putArg(gradient3)
                        .putArg(hessian00)
                        .putArg(hessian01)
                        .putArg(hessian02)
                        .putArg(hessian03)
                        .putArg(hessian11)
                        .putArg(hessian12)
                        .putArg(hessian13)
                        .putArg(hessian22)
                        .putArg(hessian23)
                        .putArg(hessian33)
                        .putArg(entryImageBuffer)
                        .putArg(maskBuffer)
                        .putNullArg(halfReductionSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putArg((int)(nrOfBlocks));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_scaledRotationSumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_scaledRotationSumInLocalMemoryCombined].rewind();
                
                buffer = queue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                area = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(maskBuffer, buffer);
                
                buffer = queue.putMapBuffer(entryImageBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                mse = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(entryImageBuffer, buffer);
                        
                buffer = queue.putMapBuffer(gradient0, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[0] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient0, buffer);
                        
                buffer = queue.putMapBuffer(gradient1, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient1, buffer);
                        
                buffer = queue.putMapBuffer(gradient2, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient2, buffer);
                
                buffer = queue.putMapBuffer(gradient3, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[3] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient3, buffer);
                        
                buffer = queue.putMapBuffer(hessian00, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][0] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian00, buffer);
                        
                buffer = queue.putMapBuffer(hessian01, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian01, buffer);
                        
                buffer = queue.putMapBuffer(hessian02, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian02, buffer);
                
                buffer = queue.putMapBuffer(hessian03, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][3] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian03, buffer);
                        
                buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[1][1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian11, buffer);
                        
                buffer = queue.putMapBuffer(hessian12, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[1][2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian12, buffer);
                
                buffer = queue.putMapBuffer(hessian13, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[1][3] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian13, buffer);
                        
                buffer = queue.putMapBuffer(hessian22, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[2][2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian22, buffer);
                
                buffer = queue.putMapBuffer(hessian23, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[2][3] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian23, buffer);
                
                buffer = queue.putMapBuffer(hessian33, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[3][3] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian33, buffer);
            }
            // symmetrize hessian
            for (int i = 1; (i < 4); i++) {
                for (int j = 0; (j < i); j++) {
                        hessian[i][j] = hessian[j][i];
                }
            }
            return mse/area;
        }
        
        //TODO: this function uses OpenCL kernels with a huge number of arguments which may fail.
        //Also a very large number of OpenCL __local buffers are used which, depending on the GPU, may cause out-of-memory errors.
        //These errors should be handled or checked beforehand.
        private double getAffineMeanSquares(int pyramidIndex, double currentoffsetx, double curentoffsety,
                double currenta11, double currenta12,
                double currenta21, double currenta22)
        {
            // First reset the global values which will not be reset in the loop
            Arrays.fill(gradient, 0.0);
            for (int i = 0; (i < 6); i++) {
                Arrays.fill(hessian[i], 0.0);
            }
            
            // now we have the diffs let's calculate all the derived elements
            int halfReductionSize = (int) (((sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)+((sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)%2))/2);
            ByteBuffer buffer;
            double area;
            // calculate the MSE row-wise
            double mse;
            if(halfReductionSize <= maximumElementsForLocalFPTcombinedSum)
            {                
                uniformBSplineTransformProgramKernels[KERNEL_affineErrorWithGradAndHess]
                        .putArg(sourcePyramid[pyramidIndex].Image)
                        .putArg(targetPyramid[pyramidIndex].Coefficient)
                        .putArg(sourcePyramid[pyramidIndex].xGradient)
                        .putArg(sourcePyramid[pyramidIndex].yGradient)
                        .putArg(maskBuffer)
                        .putArg(gradient0)
                        .putArg(gradient1)
                        .putArg(gradient2)
                        .putArg(gradient3)
                        .putArg(gradient4)
                        .putArg(gradient5)
                        .putArg(hessian00)
                        .putArg(hessian01)
                        .putArg(hessian02)
                        .putArg(hessian03)
                        .putArg(hessian04)
                        .putArg(hessian05)
                        .putArg(hessian11)
                        .putArg(hessian12)
                        .putArg(hessian13)
                        .putArg(hessian14)
                        .putArg(hessian15)
                        .putArg(hessian22)
                        .putArg(hessian23)
                        .putArg(hessian24)
                        .putArg(hessian25)
                        .putArg(hessian33)
                        .putArg(hessian34)
                        .putArg(hessian35)
                        .putArg(hessian44)
                        .putArg(hessian45)
                        .putArg(hessian55)
                        .putArg(entryImageBuffer)
                        .putArg((int)sourcePyramid[pyramidIndex].width)
                        .putArg((int)sourcePyramid[pyramidIndex].height)
                        .putArg((int)targetPyramid[pyramidIndex].width)
                        .putArg((int)targetPyramid[pyramidIndex].height)
                        .putArg((int)targetPyramid[pyramidIndex].width * 2)
                        .putArg((int)targetPyramid[pyramidIndex].height * 2);
                if(usesFloatGPU)
                {
                    uniformBSplineTransformProgramKernels[KERNEL_affineErrorWithGradAndHess]
                        .putArg((float)currentoffsetx)
                        .putArg((float)curentoffsety)
                        .putArg((float)currenta11)
                        .putArg((float)currenta12)
                        .putArg((float)currenta21)
                        .putArg((float)currenta22);
                }
                else
                {
                    uniformBSplineTransformProgramKernels[KERNEL_affineErrorWithGradAndHess]
                        .putArg(currentoffsetx)
                        .putArg(curentoffsety)
                        .putArg(currenta11)
                        .putArg(currenta12)
                        .putArg(currenta21)
                        .putArg(currenta22);
                }
                int localWorkSize = (int)Math.min(uniformBSplineTransformProgramKernels[KERNEL_affineErrorWithGradAndHess].getWorkGroupSize(device),blocksizeMultiplier*optimalMultiples[KERNEL_affineErrorWithGradAndHess]);  // Local work size dimensions
                int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_affineErrorWithGradAndHess], (int)(sourcePyramid[pyramidIndex].width*sourcePyramid[pyramidIndex].height));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_affineErrorWithGradAndHess],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_affineErrorWithGradAndHess].rewind();
                
                /*
                Because each of the following kernels is synchronized by barriers on the GPU they can only be executed on a single compute device.
                Therefore using an async queue may increase the speed by allowing multiple kernels to execute in parallel (if local memory permits)
                TODO: this is not on the async queue because the results of the first kernel are needed for the second kernel
                */
                localWorkSize = halfReductionSize % optimalMultiples[KERNEL_affineSumInLocalMemoryCombined] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_affineSumInLocalMemoryCombined] - (halfReductionSize % optimalMultiples[KERNEL_affineSumInLocalMemoryCombined])), maximumElementsForLocalFPTcombinedSum);
                globalWorkSize = localWorkSize;
                uniformBSplineTransformProgramKernels[KERNEL_affineSumInLocalMemoryCombined]
                        .putArg(maskBuffer)
                        .putArg(entryImageBuffer)
                        .putArg(gradient0)
                        .putArg(gradient1)
                        .putArg(gradient2)
                        .putArg(gradient3)
                        .putArg(gradient4)
                        .putArg(gradient5)
                        .putArg(hessian00)
                        .putArg(hessian01)
                        .putArg(hessian02)
                        .putArg(hessian03)
                        .putArg(hessian04)
                        .putArg(hessian05)
                        .putArg(hessian11)
                        .putArg(hessian12)
                        .putArg(hessian13)
                        .putArg(hessian14)
                        .putArg(hessian15)
                        .putArg(hessian22)
                        .putArg(hessian23)
                        .putArg(hessian24)
                        .putArg(hessian25)
                        .putArg(hessian33)
                        .putArg(hessian34)
                        .putArg(hessian35)
                        .putArg(hessian44)
                        .putArg(hessian45)
                        .putArg(hessian55)
                        .putNullArg(localWorkSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_affineSumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_affineSumInLocalMemoryCombined].rewind();
                
                //TODO: the implicit if's for float/double can be reduced by separating the cases into two blocks using a single if statement
                buffer = queue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                area = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(maskBuffer, buffer);
                
                buffer = queue.putMapBuffer(entryImageBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                mse = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(entryImageBuffer, buffer);
                
                buffer = queue.putMapBuffer(gradient0, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[0] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient0, buffer);
                
                buffer = queue.putMapBuffer(gradient1, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient1, buffer);
                
                buffer = queue.putMapBuffer(gradient2, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient2, buffer);
                
                buffer = queue.putMapBuffer(gradient3, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[3] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient3, buffer);
                
                buffer = queue.putMapBuffer(gradient4, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[4] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient4, buffer);
                
                buffer = queue.putMapBuffer(gradient5, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[5] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient5, buffer);
                
                buffer = queue.putMapBuffer(hessian00, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][0] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian00, buffer);
                
                buffer = queue.putMapBuffer(hessian01, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian01, buffer);
                
                buffer = queue.putMapBuffer(hessian02, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian02, buffer);
                
                buffer = queue.putMapBuffer(hessian03, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][3] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian03, buffer);
                
                buffer = queue.putMapBuffer(hessian04, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][4] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian04, buffer);
                
                buffer = queue.putMapBuffer(hessian05, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][5] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian05, buffer);
                
                buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[1][1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian11, buffer);
                
                buffer = queue.putMapBuffer(hessian12, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[1][2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian12, buffer);
                
                buffer = queue.putMapBuffer(hessian13, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[1][3] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian13, buffer);
                
                buffer = queue.putMapBuffer(hessian14, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[1][4] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian14, buffer);
                
                buffer = queue.putMapBuffer(hessian15, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[1][5] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian15, buffer);
                
                buffer = queue.putMapBuffer(hessian22, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[2][2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian22, buffer);
                
                buffer = queue.putMapBuffer(hessian23, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[2][3] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian23, buffer);
                
                buffer = queue.putMapBuffer(hessian24, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[2][4] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian24, buffer);
                
                buffer = queue.putMapBuffer(hessian25, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[2][5] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian25, buffer);
                
                buffer = queue.putMapBuffer(hessian33, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[3][3] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian33, buffer);
                
                buffer = queue.putMapBuffer(hessian34, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[3][4] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian34, buffer);
                
                buffer = queue.putMapBuffer(hessian35, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[3][5] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian35, buffer);
                
                buffer = queue.putMapBuffer(hessian44, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[4][4] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian44, buffer);
                
                buffer = queue.putMapBuffer(hessian45, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[4][5] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian45, buffer);
                
                buffer = queue.putMapBuffer(hessian55, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[5][5] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian55, buffer);
            }
            else
            {
                
                int nrOfBlocks = ((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))%blockSizesParallel;
                nrOfBlocks = nrOfBlocks == 0 ? (int)Math.min(((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))/blockSizesParallel,maximumSumReductionBlockNr) : (int)Math.min((((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))+(blockSizesParallel-nrOfBlocks))/blockSizesParallel,maximumSumReductionBlockNr);
                
                int globalWorkSize = nrOfBlocks * blockSizesParallel;
                uniformBSplineTransformProgramKernels[KERNEL_affineErrorWithGradAndHessBrent]
                        .putArg(sourcePyramid[pyramidIndex].Image)
                        .putArg(targetPyramid[pyramidIndex].Coefficient)
                        .putArg(sourcePyramid[pyramidIndex].xGradient)
                        .putArg(sourcePyramid[pyramidIndex].yGradient)
                        .putArg(gradient0)
                        .putArg(gradient1)
                        .putArg(gradient2)
                        .putArg(gradient3)
                        .putArg(gradient4)
                        .putArg(gradient5)
                        .putArg(hessian00)
                        .putArg(hessian01)
                        .putArg(hessian02)
                        .putArg(hessian03)
                        .putArg(hessian04)
                        .putArg(hessian05)
                        .putArg(hessian11)
                        .putArg(hessian12)
                        .putArg(hessian13)
                        .putArg(hessian14)
                        .putArg(hessian15)
                        .putArg(hessian22)
                        .putArg(hessian23)
                        .putArg(hessian24)
                        .putArg(hessian25)
                        .putArg(hessian33)
                        .putArg(hessian34)
                        .putArg(hessian35)
                        .putArg(hessian44)
                        .putArg(hessian45)
                        .putArg(hessian55)
                        .putArg(entryImageBuffer)
                        .putArg(maskBuffer);
                
                
                uniformBSplineTransformProgramKernels[KERNEL_affineErrorWithGradAndHessBrent]
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/);
                
                
                uniformBSplineTransformProgramKernels[KERNEL_affineErrorWithGradAndHessBrent]
                        .putArg((int)sourcePyramid[pyramidIndex].width)
                        .putArg((int)sourcePyramid[pyramidIndex].height)
                        .putArg((int)targetPyramid[pyramidIndex].width)
                        .putArg((int)targetPyramid[pyramidIndex].height);
                
                if(usesFloatGPU)
                {
                    uniformBSplineTransformProgramKernels[KERNEL_affineErrorWithGradAndHessBrent]
                        .putArg((float)currentoffsetx)
                        .putArg((float)curentoffsety)
                        .putArg((float)currenta11)
                        .putArg((float)currenta12)
                        .putArg((float)currenta21)
                        .putArg((float)currenta22);
                }
                else
                {
                    uniformBSplineTransformProgramKernels[KERNEL_affineErrorWithGradAndHessBrent]
                        .putArg(currentoffsetx)
                        .putArg(curentoffsety)
                        .putArg(currenta11)
                        .putArg(currenta12)
                        .putArg(currenta21)
                        .putArg(currenta22);
                }                
                uniformBSplineTransformProgramKernels[KERNEL_affineErrorWithGradAndHessBrent]
                        .putArg((int)targetPyramid[pyramidIndex].width*2)
                        .putArg((int)targetPyramid[pyramidIndex].height*2);
                
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_affineErrorWithGradAndHessBrent],0,globalWorkSize,blockSizesParallel);
                uniformBSplineTransformProgramKernels[KERNEL_affineErrorWithGradAndHessBrent].rewind();
                
                // use the in-memory summations (also async) each with its own buffer to do the final reduction)
                halfReductionSize = (int) ((nrOfBlocks+(nrOfBlocks%2))/2);
                int localWorkSize = halfReductionSize % optimalMultiples[KERNEL_affineSumInLocalMemoryCombined] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_affineSumInLocalMemoryCombined] - (halfReductionSize % optimalMultiples[KERNEL_affineSumInLocalMemoryCombined])), maximumElementsForLocalFPTcombinedSum);
                globalWorkSize = localWorkSize;
                uniformBSplineTransformProgramKernels[KERNEL_affineSumInLocalMemoryCombined]
                		.putArg(maskBuffer)
                        .putArg(entryImageBuffer)
                        .putArg(gradient0)
                        .putArg(gradient1)
                        .putArg(gradient2)
                        .putArg(gradient3)
                        .putArg(gradient4)
                        .putArg(gradient5)
                        .putArg(hessian00)
                        .putArg(hessian01)
                        .putArg(hessian02)
                        .putArg(hessian03)
                        .putArg(hessian04)
                        .putArg(hessian05)
                        .putArg(hessian11)
                        .putArg(hessian12)
                        .putArg(hessian13)
                        .putArg(hessian14)
                        .putArg(hessian15)
                        .putArg(hessian22)
                        .putArg(hessian23)
                        .putArg(hessian24)
                        .putArg(hessian25)
                        .putArg(hessian33)
                        .putArg(hessian34)
                        .putArg(hessian35)
                        .putArg(hessian44)
                        .putArg(hessian45)
                        .putArg(hessian55)
                        .putNullArg(halfReductionSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putArg((int)(nrOfBlocks));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_affineSumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_affineSumInLocalMemoryCombined].rewind();
                
                buffer = queue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                area = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(maskBuffer, buffer);
                
                buffer = queue.putMapBuffer(entryImageBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                mse = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(entryImageBuffer, buffer);
                        
                buffer = queue.putMapBuffer(gradient0, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[0] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient0, buffer);
                        
                buffer = queue.putMapBuffer(gradient1, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient1, buffer);
                        
                buffer = queue.putMapBuffer(gradient2, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient2, buffer);
                
                buffer = queue.putMapBuffer(gradient3, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[3] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient3, buffer);
                
                buffer = queue.putMapBuffer(gradient4, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[4] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient4, buffer);
                
                buffer = queue.putMapBuffer(gradient5, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                gradient[5] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(gradient5, buffer);
                        
                buffer = queue.putMapBuffer(hessian00, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][0] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian00, buffer);
                        
                buffer = queue.putMapBuffer(hessian01, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian01, buffer);
                        
                buffer = queue.putMapBuffer(hessian02, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian02, buffer);
                
                buffer = queue.putMapBuffer(hessian03, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][3] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian03, buffer);
                
                buffer = queue.putMapBuffer(hessian04, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][4] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian04, buffer);
                
                buffer = queue.putMapBuffer(hessian05, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[0][5] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian05, buffer);
                        
                buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[1][1] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian11, buffer);
                        
                buffer = queue.putMapBuffer(hessian12, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[1][2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian12, buffer);
                
                buffer = queue.putMapBuffer(hessian13, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[1][3] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian13, buffer);
                
                buffer = queue.putMapBuffer(hessian14, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[1][4] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian14, buffer);
                
                buffer = queue.putMapBuffer(hessian15, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[1][5] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian15, buffer);
                        
                buffer = queue.putMapBuffer(hessian22, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[2][2] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian22, buffer);
                
                buffer = queue.putMapBuffer(hessian23, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[2][3] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian23, buffer);
                
                buffer = queue.putMapBuffer(hessian24, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[2][4] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian24, buffer);
                
                buffer = queue.putMapBuffer(hessian25, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[2][5] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian25, buffer);
                
                buffer = queue.putMapBuffer(hessian33, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[3][3] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian33, buffer);
                
                buffer = queue.putMapBuffer(hessian34, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[3][4] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian34, buffer);
                
                buffer = queue.putMapBuffer(hessian35, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[3][5] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian35, buffer);
                
                buffer = queue.putMapBuffer(hessian44, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[4][4] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian44, buffer);
                
                buffer = queue.putMapBuffer(hessian45, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[4][5] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian45, buffer);
                
                buffer = queue.putMapBuffer(hessian55, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                hessian[5][5] = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                queue.putUnmapMemory(hessian55, buffer);
            }
            // symmetrize hessian
            for (int i = 1; (i < 6); i++) {
                for (int j = 0; (j < i); j++) {
                        hessian[i][j] = hessian[j][i];
                }
            }
            return mse/area;
        }
        
        private double getTranslationMeanSquaresWithoutHessian(int pyramidIndex, double currentoffsetx, double curentoffsety)
        {
            uniformBSplineTransformProgramKernels[KERNEL_translationError]
                    .putArg(sourcePyramid[pyramidIndex].Image)
                    .putArg(targetPyramid[pyramidIndex].Coefficient)
                    .putArg(entryImageBuffer)
                    .putArg(maskBuffer)
                    .putArg((int)sourcePyramid[pyramidIndex].width)
                    .putArg((int)sourcePyramid[pyramidIndex].height)
                    .putArg((int)targetPyramid[pyramidIndex].width)
                    .putArg((int)targetPyramid[pyramidIndex].height)
                    .putArg((int)targetPyramid[pyramidIndex].width * 2)
                    .putArg((int)targetPyramid[pyramidIndex].height * 2);
            
            if (usesFloatGPU) {
                uniformBSplineTransformProgramKernels[KERNEL_translationError]
                        .putArg((float)currentoffsetx)
                        .putArg((float)curentoffsety);
            } else {
                uniformBSplineTransformProgramKernels[KERNEL_translationError]
                        .putArg(currentoffsetx)
                        .putArg(curentoffsety);
            }           
            int localWorkSize = (int)Math.min(uniformBSplineTransformProgramKernels[KERNEL_translationError].getWorkGroupSize(device),optimalMultiples[KERNEL_translationError]*blocksizeMultiplier);  // Local work size dimensions
            int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_translationError], (int)(sourcePyramid[pyramidIndex].width*sourcePyramid[pyramidIndex].height));
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_translationError],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_translationError].rewind();
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
                        //TODO/FIXME: should the local buffer be halfReductionSize or localWorkSize in size?
                        .putNullArg(halfReductionSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/)
                        .putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height));
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].putArg(entryImageBuffer).putNullArg(halfReductionSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/).putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height));
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                asyncQueue.finish();
                buffer = asyncQueue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                area = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                asyncQueue.putUnmapMemory(maskBuffer, buffer);
                buffer = asyncQueue.putMapBuffer(entryImageBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                mse = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                asyncQueue.putUnmapMemory(entryImageBuffer, buffer);
                asyncQueue.finish();
            }
            else
            {
                nrOfBlocks = ((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))%blockSizesFPT;
                nrOfBlocks = nrOfBlocks == 0 ? (int)Math.min(((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))/blockSizesFPT,maximumSumReductionBlockNr) : (int)Math.min((((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))+(blockSizesFPT-nrOfBlocks))/blockSizesFPT,maximumSumReductionBlockNr);
                
                globalWorkSize = nrOfBlocks * blockSizesFPT;
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].putArg(maskBuffer).putArg(parallelSumReductionBuffers[0]).putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)).putNullArg(blockSizesFPT*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction],0,globalWorkSize,blockSizesFPT);
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].rewind();
                
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].putArg(entryImageBuffer).putArg(parallelSumReductionBuffers[1]).putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)).putNullArg(blockSizesFPT*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction],0,globalWorkSize,blockSizesFPT);
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].rewind();
                
                // synchronize the data
                asyncQueue.finish();
                
                // reduce in local memory
                halfReductionSize = (int) ((nrOfBlocks+(nrOfBlocks%2))/2);
                localWorkSize = halfReductionSize % optimalMultiples[KERNEL_sumInLocalMemory] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_sumInLocalMemory] - (halfReductionSize % optimalMultiples[KERNEL_sumInLocalMemory])), maximumElementsForLocalFPTsum);
                globalWorkSize = localWorkSize;
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].putArg(parallelSumReductionBuffers[0]).putNullArg(halfReductionSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/).putArg(nrOfBlocks);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].putArg(parallelSumReductionBuffers[1]).putNullArg(halfReductionSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/).putArg(nrOfBlocks);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                
                // synchronize again
                asyncQueue.finish();
                
                // Download data
                buffer = asyncQueue.putMapBuffer(parallelSumReductionBuffers[0], CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                area = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                asyncQueue.putUnmapMemory(parallelSumReductionBuffers[0], buffer);
                buffer = asyncQueue.putMapBuffer(parallelSumReductionBuffers[1], CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                mse = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                asyncQueue.putUnmapMemory(parallelSumReductionBuffers[1], buffer);
                asyncQueue.finish();
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
                    .putArg((int)targetPyramid[pyramidIndex].width * 2)
                    .putArg((int)targetPyramid[pyramidIndex].height * 2);
            
            if (usesFloatGPU) {
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyError]
                    .putArg((float)currentoffsetx)
                    .putArg((float)curentoffsety)
                    .putArg((float)Math.cos(currentangle))
                    .putArg((float)-Math.sin(currentangle));
            } else {
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyError]
                    .putArg(currentoffsetx)
                    .putArg(curentoffsety)
                    .putArg(Math.cos(currentangle))
                    .putArg(-Math.sin(currentangle));
            }           
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
                
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].putArg(maskBuffer).putNullArg(halfReductionSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/).putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height));
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].putArg(entryImageBuffer).putNullArg(halfReductionSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/).putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height));
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                asyncQueue.finish();
                buffer = asyncQueue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                area = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                asyncQueue.putUnmapMemory(maskBuffer, buffer);
                buffer = asyncQueue.putMapBuffer(entryImageBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                mse = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                asyncQueue.putUnmapMemory(entryImageBuffer, buffer);
                asyncQueue.finish();
            }
            else
            {
                nrOfBlocks = ((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))%blockSizesFPT;
                nrOfBlocks = nrOfBlocks == 0 ? (int)Math.min(((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))/blockSizesFPT,maximumSumReductionBlockNr) : (int)Math.min((((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))+(blockSizesFPT-nrOfBlocks))/blockSizesFPT,maximumSumReductionBlockNr);
                
                globalWorkSize = nrOfBlocks * blockSizesFPT;
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].putArg(maskBuffer).putArg(parallelSumReductionBuffers[0]).putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)).putNullArg(blockSizesFPT*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction],0,globalWorkSize,blockSizesFPT);
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].rewind();
                
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].putArg(entryImageBuffer).putArg(parallelSumReductionBuffers[1]).putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)).putNullArg(blockSizesFPT*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction],0,globalWorkSize,blockSizesFPT);
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].rewind();
                
                // synchronize the data
                asyncQueue.finish();
                
                // reduce in local memory
                halfReductionSize = (int) ((nrOfBlocks+(nrOfBlocks%2))/2);
                localWorkSize = halfReductionSize % optimalMultiples[KERNEL_sumInLocalMemory] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_sumInLocalMemory] - (halfReductionSize % optimalMultiples[KERNEL_sumInLocalMemory])), maximumElementsForLocalFPTsum);
                globalWorkSize = localWorkSize;
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].putArg(parallelSumReductionBuffers[0]).putNullArg(halfReductionSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/).putArg(nrOfBlocks);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].putArg(parallelSumReductionBuffers[1]).putNullArg(halfReductionSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/).putArg(nrOfBlocks);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                
                // synchronize again
                asyncQueue.finish();
                
                // Download data
                buffer = asyncQueue.putMapBuffer(parallelSumReductionBuffers[0], CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                area = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                asyncQueue.putUnmapMemory(parallelSumReductionBuffers[0], buffer);
                buffer = asyncQueue.putMapBuffer(parallelSumReductionBuffers[1], CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                mse = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                asyncQueue.putUnmapMemory(parallelSumReductionBuffers[1], buffer);
                asyncQueue.finish();
            }
            return mse/area;
        }
        
        private double getScaledRotationMeanSquaresWithoutHessian(int pyramidIndex, double currentoffsetx, double curentoffsety, double currentangle, double scale)
        {
            uniformBSplineTransformProgramKernels[KERNEL_scaledRotationError]
                    .putArg(sourcePyramid[pyramidIndex].Image)
                    .putArg(targetPyramid[pyramidIndex].Coefficient)
                    .putArg(entryImageBuffer)
                    .putArg(maskBuffer)
                    .putArg((int)sourcePyramid[pyramidIndex].width)
                    .putArg((int)sourcePyramid[pyramidIndex].height)
                    .putArg((int)targetPyramid[pyramidIndex].width)
                    .putArg((int)targetPyramid[pyramidIndex].height)
                    .putArg((int)targetPyramid[pyramidIndex].width * 2)
                    .putArg((int)targetPyramid[pyramidIndex].height * 2);
            
            if (usesFloatGPU) {
                uniformBSplineTransformProgramKernels[KERNEL_scaledRotationError]
                    .putArg((float)currentoffsetx)
                    .putArg((float)curentoffsety)
                    .putArg((float)Math.cos(currentangle))
                    .putArg((float)-Math.sin(currentangle))
                    .putArg((float)scale);
            } else {
                uniformBSplineTransformProgramKernels[KERNEL_scaledRotationError]
                    .putArg(currentoffsetx)
                    .putArg(curentoffsety)
                    .putArg(Math.cos(currentangle))
                    .putArg(-Math.sin(currentangle))
                    .putArg(scale);
            }           
            int localWorkSize = (int)Math.min(uniformBSplineTransformProgramKernels[KERNEL_scaledRotationError].getWorkGroupSize(device),optimalMultiples[KERNEL_scaledRotationError]*blocksizeMultiplier);  // Local work size dimensions
            int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_scaledRotationError], (int)(sourcePyramid[pyramidIndex].width*sourcePyramid[pyramidIndex].height));
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_scaledRotationError],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_scaledRotationError].rewind();
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
                
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].putArg(maskBuffer).putNullArg(halfReductionSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/).putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height));
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].putArg(entryImageBuffer).putNullArg(halfReductionSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/).putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height));
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                asyncQueue.finish();
                buffer = asyncQueue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                area = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                asyncQueue.putUnmapMemory(maskBuffer, buffer);
                buffer = asyncQueue.putMapBuffer(entryImageBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                mse = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                asyncQueue.putUnmapMemory(entryImageBuffer, buffer);
                asyncQueue.finish();
            }
            else
            {
                nrOfBlocks = ((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))%blockSizesFPT;
                nrOfBlocks = nrOfBlocks == 0 ? (int)Math.min(((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))/blockSizesFPT,maximumSumReductionBlockNr) : (int)Math.min((((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))+(blockSizesFPT-nrOfBlocks))/blockSizesFPT,maximumSumReductionBlockNr);
                
                globalWorkSize = nrOfBlocks * blockSizesFPT;
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].putArg(maskBuffer).putArg(parallelSumReductionBuffers[0]).putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)).putNullArg(blockSizesFPT*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction],0,globalWorkSize,blockSizesFPT);
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].rewind();
                
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].putArg(entryImageBuffer).putArg(parallelSumReductionBuffers[1]).putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)).putNullArg(blockSizesFPT*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction],0,globalWorkSize,blockSizesFPT);
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].rewind();
                
                // synchronize the data
                asyncQueue.finish();
                
                // reduce in local memory
                halfReductionSize = (int) ((nrOfBlocks+(nrOfBlocks%2))/2);
                localWorkSize = halfReductionSize % optimalMultiples[KERNEL_sumInLocalMemory] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_sumInLocalMemory] - (halfReductionSize % optimalMultiples[KERNEL_sumInLocalMemory])), maximumElementsForLocalFPTsum);
                globalWorkSize = localWorkSize;
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].putArg(parallelSumReductionBuffers[0]).putNullArg(halfReductionSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/).putArg(nrOfBlocks);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].putArg(parallelSumReductionBuffers[1]).putNullArg(halfReductionSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/).putArg(nrOfBlocks);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                
                // synchronize again
                asyncQueue.finish();
                
                // Download data
                buffer = asyncQueue.putMapBuffer(parallelSumReductionBuffers[0], CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                area = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                asyncQueue.putUnmapMemory(parallelSumReductionBuffers[0], buffer);
                buffer = asyncQueue.putMapBuffer(parallelSumReductionBuffers[1], CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                mse = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                asyncQueue.putUnmapMemory(parallelSumReductionBuffers[1], buffer);
                asyncQueue.finish();
            }
            return mse/area;
        }
        
        private double getAffineMeanSquaresWithoutHessian(int pyramidIndex, double currentoffsetx, double curentoffsety,
                double currenta11, double currenta12,
                double currenta21, double currenta22)
        {
            uniformBSplineTransformProgramKernels[KERNEL_affineError]
                    .putArg(sourcePyramid[pyramidIndex].Image)
                    .putArg(targetPyramid[pyramidIndex].Coefficient)
                    .putArg(entryImageBuffer)
                    .putArg(maskBuffer)
                    .putArg((int)sourcePyramid[pyramidIndex].width)
                    .putArg((int)sourcePyramid[pyramidIndex].height)
                    .putArg((int)targetPyramid[pyramidIndex].width)
                    .putArg((int)targetPyramid[pyramidIndex].height)
                    .putArg((int)targetPyramid[pyramidIndex].width * 2)
                    .putArg((int)targetPyramid[pyramidIndex].height * 2);
            
            if (usesFloatGPU) {
                uniformBSplineTransformProgramKernels[KERNEL_affineError]
                    .putArg((float)currentoffsetx)
                    .putArg((float)curentoffsety)
                    .putArg((float)currenta11)
                    .putArg((float)currenta12)
                    .putArg((float)currenta21)
                    .putArg((float)currenta22);
            } else {
                uniformBSplineTransformProgramKernels[KERNEL_affineError]
                    .putArg(currentoffsetx)
                    .putArg(curentoffsety)
                    .putArg(currenta11)
                    .putArg(currenta12)
                    .putArg(currenta21)
                    .putArg(currenta22);
            }           
            int localWorkSize = (int)Math.min(uniformBSplineTransformProgramKernels[KERNEL_affineError].getWorkGroupSize(device),optimalMultiples[KERNEL_affineError]*blocksizeMultiplier);  // Local work size dimensions
            int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_affineError], (int)(sourcePyramid[pyramidIndex].width*sourcePyramid[pyramidIndex].height));
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_scaledRotationError],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_affineError].rewind();
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
                
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].putArg(maskBuffer).putNullArg(halfReductionSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/).putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height));
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].putArg(entryImageBuffer).putNullArg(halfReductionSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/).putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height));
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                asyncQueue.finish();
                buffer = asyncQueue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                area = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                asyncQueue.putUnmapMemory(maskBuffer, buffer);
                buffer = asyncQueue.putMapBuffer(entryImageBuffer, CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                mse = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                asyncQueue.putUnmapMemory(entryImageBuffer, buffer);
                asyncQueue.finish();
            }
            else
            {
                nrOfBlocks = ((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))%blockSizesFPT;
                nrOfBlocks = nrOfBlocks == 0 ? (int)Math.min(((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))/blockSizesFPT,maximumSumReductionBlockNr) : (int)Math.min((((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))+(blockSizesFPT-nrOfBlocks))/blockSizesFPT,maximumSumReductionBlockNr);
                
                globalWorkSize = nrOfBlocks * blockSizesFPT;
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].putArg(maskBuffer).putArg(parallelSumReductionBuffers[0]).putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)).putNullArg(blockSizesFPT*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction],0,globalWorkSize,blockSizesFPT);
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].rewind();
                
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].putArg(entryImageBuffer).putArg(parallelSumReductionBuffers[1]).putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height)).putNullArg(blockSizesFPT*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction],0,globalWorkSize,blockSizesFPT);
                uniformBSplineTransformProgramKernels[KERNEL_parallelGroupedSumReduction].rewind();
                
                // synchronize the data
                asyncQueue.finish();
                
                // reduce in local memory
                halfReductionSize = (int) ((nrOfBlocks+(nrOfBlocks%2))/2);
                localWorkSize = halfReductionSize % optimalMultiples[KERNEL_sumInLocalMemory] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_sumInLocalMemory] - (halfReductionSize % optimalMultiples[KERNEL_sumInLocalMemory])), maximumElementsForLocalFPTsum);
                globalWorkSize = localWorkSize;
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].putArg(parallelSumReductionBuffers[0]).putNullArg(halfReductionSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/).putArg(nrOfBlocks);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].putArg(parallelSumReductionBuffers[1]).putNullArg(halfReductionSize*(usesFloatGPU?4:8) /*size in bytes of local mem allocation*/).putArg(nrOfBlocks);
                asyncQueue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemory].rewind();
                
                // synchronize again
                asyncQueue.finish();
                
                // Download data
                buffer = asyncQueue.putMapBuffer(parallelSumReductionBuffers[0], CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                area = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                asyncQueue.putUnmapMemory(parallelSumReductionBuffers[0], buffer);
                buffer = asyncQueue.putMapBuffer(parallelSumReductionBuffers[1], CLMemory.Map.READ, 0, (usesFloatGPU?4:8), true);
                buffer.rewind();
                mse = (usesFloatGPU?(double)buffer.asFloatBuffer().get():(double)buffer.asDoubleBuffer().get());
                asyncQueue.putUnmapMemory(parallelSumReductionBuffers[1], buffer);
                asyncQueue.finish();
            }
            return mse/area;
        }
        
        private void constructSourceImagePyramid()
        {
            // TODO: use localWorkGroup size in a more sensible manner
            int width = (int)sharedContext.img.dimension(0);
            int height = (int)sharedContext.img.dimension(1);
            int localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].getWorkGroupSize(device);  // Local work size dimensions
            int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2Dpremulhp], width*height);   // rounded up to the nearest multiple of the localWorkSize
            // pre-multiply the image
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].putArg(entryImageBuffer).putArg(width*height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp].putArg(entryImageBuffer).putArg(width).putArg(height);
            // Conversion to B-spline coefficients along X axis (Group size must be >= height)
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DXhp], height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp],0,globalWorkSize,localWorkSize);
            // X-derivatives
            uniformBSplineTransformProgramKernels[KERNEL_antiSymmetricFirMirrorOffBounds1DX].putArg(entryImageBuffer).putArg(sourcePyramid[0].xGradient).putArg(width).putArg(height);
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_antiSymmetricFirMirrorOffBounds1DX].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_antiSymmetricFirMirrorOffBounds1DX], width*height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_antiSymmetricFirMirrorOffBounds1DX],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_antiSymmetricFirMirrorOffBounds1DX].rewind();
            // Now along Y-axis
            // Has to be pre-multiplied by lambda again!!! (the kernel is still setup correctly)
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2Dpremulhp], width*height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp].putArg(entryImageBuffer).putArg(width).putArg(height);
            // Conversion to B-spline coefficients along Y axis (Group size must be >= width)
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DYhp], width);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp].rewind();
            // No need to copy the B-spline coefficients for the source, because only the images are required and that was already copied during the conversion step
            
            // The Y-derivatives still need to be calculated from the Y-coefficients
            // First calculate the Y-coefficients and only the Y-coefficients
            // Has to be pre-multiplied by lambda again!!! avoid copying data so use an out-of-place modifying calculation
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_TargetedCubicBSplinePrefilter2Dpremulhp].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_TargetedCubicBSplinePrefilter2Dpremulhp], width*height);
            uniformBSplineTransformProgramKernels[KERNEL_TargetedCubicBSplinePrefilter2Dpremulhp].putArg(sourcePyramid[0].Image).putArg(fullSizedGPUResidentHelperBuffer).putArg(width*height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_TargetedCubicBSplinePrefilter2Dpremulhp],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_TargetedCubicBSplinePrefilter2Dpremulhp].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp].putArg(fullSizedGPUResidentHelperBuffer).putArg(width).putArg(height);
            // Conversion to B-spline coefficients along Y axis (Group size must be >= width)
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DYhp], width);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp],0,globalWorkSize,localWorkSize);

            // calculate the derivatives
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_antiSymmetricFirMirrorOffBounds1DY].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_antiSymmetricFirMirrorOffBounds1DY], width*height);
            uniformBSplineTransformProgramKernels[KERNEL_antiSymmetricFirMirrorOffBounds1DY].putArg(fullSizedGPUResidentHelperBuffer).putArg(sourcePyramid[0].yGradient).putArg(width).putArg(height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_antiSymmetricFirMirrorOffBounds1DY],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_antiSymmetricFirMirrorOffBounds1DY].rewind();
            // prepare the image for resampling by applying the FIR filter of degree 7 (out of place mod)
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhpDeg7].putArg(entryImageBuffer).putArg(fullSizedGPUResidentHelperBuffer).putArg(width).putArg(height);
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhpDeg7].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_BasicToCardinal2DXhpDeg7], width*height);
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
                uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DY].putArg(fullSizedGPUResidentHelperBuffer).putArg(sourcePyramid[j].Image).putArg(width/2).putArg(height).putArg(height/2);
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DY].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_reduceDual1DY], ((int)(width/2))*((int)(height/2)));// Warning: integer division don't change
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DY],0,globalWorkSize,localWorkSize);
                if(j < pyramidDepth - 1)
                {
                    uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX].rewind();//reset argument index
                    uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX].putArg(sourcePyramid[j].Image).putArg(fullSizedGPUResidentHelperBuffer).putArg(width/2).putArg(height/2).putArg(((int)(width/2))/2 /*Warning integer division don't change*/ );
                    localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX].getWorkGroupSize(device);
                    globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_reduceDual1DX], ((int)(((int)(width/2))/2))*((int)(height/2)));// Warning: integer division don't change
                    queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX],0,globalWorkSize,localWorkSize);
                }
                width /= 2;
                height /= 2;
                // restore the B-spline coefficients
                uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp].putArg(sourcePyramid[j].Image).putArg(width*height);
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DDeg7premulhp], width*height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXDeg7hp].putArg(sourcePyramid[j].Image).putArg(width).putArg(height);
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXDeg7hp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DXDeg7hp], height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXDeg7hp],0,globalWorkSize,localWorkSize);
                // pre-multiply again (the kernel is still setup correctly)
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DDeg7premulhp], width*height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYDeg7hp].putArg(sourcePyramid[j].Image).putArg(width).putArg(height);
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYDeg7hp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DYDeg7hp], width);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYDeg7hp],0,globalWorkSize,localWorkSize);
                // Now we have the restored downsampled coefficients still need to calculate the derivatives
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
                // For the source image the downsampled image has to be restored from the B-spline coefficients and to avoid wasting memory the entry buffer will be used as intermediate buffer
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
                uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXDeg7hp].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYDeg7hp].rewind();
            }
            uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhpDeg7].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhpDeg7].rewind();
        }
        private void constructTargetImagePyramid()
        {
            // TODO: use localWorkGroup size in a more sensible manner
            int width = (int)sharedContext.img.dimension(0);
            int height = (int)sharedContext.img.dimension(1);
            int localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_TargetedCubicBSplinePrefilter2Dpremulhp].getWorkGroupSize(device);  // Local work size dimensions
            int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_TargetedCubicBSplinePrefilter2Dpremulhp], width*height);   // rounded up to the nearest multiple of the localWorkSize
            // pre-multiply the image
            uniformBSplineTransformProgramKernels[KERNEL_TargetedCubicBSplinePrefilter2Dpremulhp].putArg(entryImageBuffer).putArg(targetPyramid[0].Coefficient).putArg(width*height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_TargetedCubicBSplinePrefilter2Dpremulhp],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_TargetedCubicBSplinePrefilter2Dpremulhp].rewind();
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp].getWorkGroupSize(device);
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp].putArg(targetPyramid[0].Coefficient).putArg(width).putArg(height);
            // Conversion to B-spline coefficients along X axis (Group size must be >= height)
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DXhp], height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp],0,globalWorkSize,localWorkSize);
            // Now along the Y-axis
            // Has to be pre-multiplied by lambda again!!!
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].putArg(targetPyramid[0].Coefficient).putArg(width*height);
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2Dpremulhp], width*height);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp].putArg(targetPyramid[0].Coefficient).putArg(width).putArg(height);
            // Conversion to B-spline coefficients along Y axis (Group size must be >= width)
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DYhp], width);
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp],0,globalWorkSize,localWorkSize);
            // prepare the image for resampling by applying the FIR filter of degree 7 (out of place mod)
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhpDeg7].putArg(targetPyramid[0].Coefficient).putArg(fullSizedGPUResidentHelperBuffer).putArg(width).putArg(height);
            localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhpDeg7].getWorkGroupSize(device);
            globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_BasicToCardinal2DXhpDeg7], width*height);
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
                uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXDeg7hp].putArg(targetPyramid[j].Coefficient).putArg(width).putArg(height);
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXDeg7hp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DXDeg7hp], height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXDeg7hp],0,globalWorkSize,localWorkSize);
                // pre-multiply again (the kernel is still set up correctly)
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DDeg7premulhp], width*height);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYDeg7hp].putArg(targetPyramid[j].Coefficient).putArg(width).putArg(height);
                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYDeg7hp].getWorkGroupSize(device);
                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_CubicBSplinePrefilter2DYDeg7hp], width);
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYDeg7hp],0,globalWorkSize,localWorkSize);
                // Now we have the restored downsampled coefficients and they are already in the pyramid storage so no need to copy them just rewind the argument queues
                uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DY].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXDeg7hp].rewind();
                uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYDeg7hp].rewind();
            }
            uniformBSplineTransformProgramKernels[KERNEL_reduceDual1DX].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2Dpremulhp].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXhp].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYhp].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhpDeg7].rewind();
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhpDeg7].rewind();
        }
        private void putSourceImageIntoPipelineEntry()
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
            } else if ((sharedContext.img.firstElement() instanceof LongType)||(sharedContext.img.firstElement() instanceof UnsignedLongType)) {
                ByteBuffer buffer = queue.putMapBuffer(conversionEntryBuffer, CLMemory.Map.WRITE, true);
                buffer.rewind();
                buffer.asLongBuffer().put((long[]) scat.sourceArray);
                queue.putUnmapMemory(conversionEntryBuffer, buffer);
            } else if ((sharedContext.img.firstElement() instanceof FloatType)) {
                if (!usesFloatGPU) {
                    // Conversion buffers are only needed if the representation is double later on
                    ByteBuffer buffer = queue.putMapBuffer(conversionEntryBuffer, CLMemory.Map.WRITE, true);
                    buffer.rewind();
                    buffer.asFloatBuffer().put((float[]) scat.sourceArray);
                    queue.putUnmapMemory(conversionEntryBuffer, buffer);
                } else {
                    // otherwise put it directly into the image buffer
                    ByteBuffer buffer = queue.putMapBuffer(entryImageBuffer, CLMemory.Map.WRITE, true);
                    buffer.rewind();
                    buffer.asFloatBuffer().put((float[]) scat.sourceArray);
                    queue.putUnmapMemory(entryImageBuffer, buffer);
                    queue.putCopyBuffer(entryImageBuffer,sourcePyramid[0].Image);// Copy the first image to the pyramid because it would have to be moved again later
                    return;
                }
            } else if ((sharedContext.img.firstElement() instanceof DoubleType)) {
                // Due to the pretest this makes usesFloatGPU = false => no conversion buffer is needed
                // put the image directly into the image entry buffer
                ByteBuffer buffer = queue.putMapBuffer(entryImageBuffer, CLMemory.Map.WRITE, true);
                buffer.rewind();
                buffer.asDoubleBuffer().put((double[]) scat.sourceArray);
                queue.putUnmapMemory(entryImageBuffer, buffer);
                queue.putCopyBuffer(entryImageBuffer,sourcePyramid[0].Image);// Copy the first image to the pyramid because it would have to be moved again later
                return;
            }
            int localWorkSize = (int)conversionProgramKernel.getWorkGroupSize(device);  // Local work size dimensions
            int globalWorkSize = StaticUtility.roundUp(localWorkSize, conversionProgramKernel.getPreferredWorkGroupSizeMultiple(device), (int)sharedContext.img.dimension(0)*(int)sharedContext.img.dimension(1));   // rounded up to the nearest multiple of the localWorkSize
            queue.put1DRangeKernel(conversionProgramKernel,0,globalWorkSize,localWorkSize);
            queue.putCopyBuffer(entryImageBuffer,sourcePyramid[0].Image);// Copy the first image to the pyramid because it would have to be moved again later
        }
        private void putTargetImageIntoPipelineEntry()
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
            } else if ((sharedContext.img.firstElement() instanceof LongType)||(sharedContext.img.firstElement() instanceof UnsignedLongType)) {
                ByteBuffer buffer = queue.putMapBuffer(conversionEntryBuffer, CLMemory.Map.WRITE, true);
                buffer.rewind();
                buffer.asLongBuffer().put((long[]) scat.targetArray);
                queue.putUnmapMemory(conversionEntryBuffer, buffer);
            } else if ((sharedContext.img.firstElement() instanceof FloatType)) {
                if (!usesFloatGPU) {
                    // Conversion buffers are only needed if the representation is double later on
                    ByteBuffer buffer = queue.putMapBuffer(conversionEntryBuffer, CLMemory.Map.WRITE, true);
                    buffer.rewind();
                    buffer.asFloatBuffer().put((float[]) scat.targetArray);
                    queue.putUnmapMemory(conversionEntryBuffer, buffer);
                } else {
                    // otherwise put it directly into the image buffer
                    ByteBuffer buffer = queue.putMapBuffer(entryImageBuffer, CLMemory.Map.WRITE, true);
                    buffer.rewind();
                    buffer.asFloatBuffer().put((float[]) scat.targetArray);
                    queue.putUnmapMemory(entryImageBuffer, buffer);
                    return;
                }
            } else if ((sharedContext.img.firstElement() instanceof DoubleType)) {
                // Due to the pretest this makes usesFloatGPU = false => no conversion buffer is needed
                // put the image directly into the image entry buffer
                ByteBuffer buffer = queue.putMapBuffer(entryImageBuffer, CLMemory.Map.WRITE, true);
                buffer.rewind();
                buffer.asDoubleBuffer().put((double[]) scat.targetArray);
                queue.putUnmapMemory(entryImageBuffer, buffer);
                return;
            }
            int localWorkSize = (int)conversionProgramKernel.getWorkGroupSize(device);  // Local work size dimensions
            int globalWorkSize = StaticUtility.roundUp(localWorkSize, conversionProgramKernel.getPreferredWorkGroupSizeMultiple(device),(int) sharedContext.img.dimension(0)*(int)sharedContext.img.dimension(1));   // rounded up to the nearest multiple of the localWorkSize
            queue.put1DRangeKernel(conversionProgramKernel,0,globalWorkSize,localWorkSize);
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

    public OCLNGStackReg(boolean useFloatGPUOnly, final AbstractSharedContext sharedContext) throws Exception
    {
        super(sharedContext);
        if(this.sharedContext.forceDoublePrecisionRepr)
        {
            this.useFloatGPUOnly = false;
        }
        else
        {
            this.useFloatGPUOnly = useFloatGPUOnly;
        }
        calculatePyramidDepth();
        enumerateOCLDevicesAndInitialize();
        sharedContext.addParties(workers.length);
    }
    private void enumerateOCLDevicesAndInitialize() throws Exception
    {
        /*
        Created command queues in in-order execution by not setting out-of-order
        execution. According to the docs this allows one to call multiple kernels
        after one another without cross synchronization
        */
        if(this.useFloatGPUOnly)
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
            usesFloatGPU = true;
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
        workers = new OCLNGStackRegWorker[nrOfWorkers];
        for(int i = 0; i < nrOfWorkers; i++)
        {
            workers[i] = new OCLNGStackRegWorker(contexts[i],devices[i]);
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
        for(OCLNGStackRegWorker worker: workers)
        {
            worker.start();
        }
    }
    
    @Override
    public void waitForFinish()  throws InterruptedException {
        if(workers != null)
        {
            for(OCLNGStackRegWorker worker: workers)
            {
                if(worker != null)
                {
                    worker.getThread().join();
                }
            }
        }
    }
}
