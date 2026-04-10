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
import ch.unibas.biozentrum.imagejplugins.util.AffineTransformation;
import ch.unibas.biozentrum.imagejplugins.util.RigidBodyTransformation;
import ch.unibas.biozentrum.imagejplugins.util.ScaledRotationTransformation;
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
public class HybridPrecisionNGStackReg extends RegistrationAndTransformation
{
    // Just constants for accessing the compiled kernels in the program (compiled separately for each device)
    private static final int NR_OF_OPENCL_KERNELS = 35;
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
    private static final int KERNEL_dresizingTransformImageWithBsplineInterpolation = 34;
    
    private static final int KERNEL_ftranslationError = 14; //KERNEL_rigidBodyError
    private static final int KERNEL_ftranslationErrorWithGradAndHess = 15; //KERNEL_rigidBodyErrorWithGradAndHess
    private static final int KERNEL_ftranslationSumInLocalMemoryCombined = 18; //KERNEL_sumInLocalMemoryCombined
    private static final int KERNEL_ftranslationErrorWithGradAndHessBrent = 19; //KERNEL_rigidBodyErrorWithGradAndHessBrent
    
    private static final int KERNEL_dtranslationError = 27; //KERNEL_drigidBodyError
    private static final int KERNEL_dtranslationErrorWithGradAndHess = 28; //KERNEL_drigidBodyErrorWithGradAndHess
    private static final int KERNEL_dtranslationSumInLocalMemoryCombined = 30; //KERNEL_dsumInLocalMemoryCombined
    private static final int KERNEL_dtranslationErrorWithGradAndHessBrent = 32; //KERNEL_drigidBodyErrorWithGradAndHessBrent
    private static final int KERNEL_dtranslationtransformImageWithBsplineInterpolation = 33; //KERNEL_dtransformImageWithBsplineInterpolation
    
    private static final int KERNEL_fscaledRotationError = 14; //KERNEL_rigidBodyError
    private static final int KERNEL_fscaledRotationErrorWithGradAndHess = 15; //KERNEL_rigidBodyErrorWithGradAndHess
    private static final int KERNEL_fscaledRotationSumInLocalMemoryCombined = 18; //KERNEL_sumInLocalMemoryCombined
    private static final int KERNEL_fscaledRotationErrorWithGradAndHessBrent = 19; //KERNEL_rigidBodyErrorWithGradAndHessBrent
    
    private static final int KERNEL_dscaledRotationError = 27;
    private static final int KERNEL_dscaledRotationErrorWithGradAndHess = 28;
    private static final int KERNEL_dscaledRotationErrorWithGradAndHessBrent = 32;
    private static final int KERNEL_dscaledRotationTransformImageWithBsplineInterpolation = 33;
    private static final int KERNEL_dscaledRotationSumInLocalMemoryCombined = 30;
    
    private static final int KERNEL_faffineError = 14; //KERNEL_rigidBodyError
    private static final int KERNEL_faffineErrorWithGradAndHess = 15; //KERNEL_rigidBodyErrorWithGradAndHess
    private static final int KERNEL_faffineSumInLocalMemoryCombined = 18; //KERNEL_sumInLocalMemoryCombined
    private static final int KERNEL_faffineErrorWithGradAndHessBrent = 19; //KERNEL_rigidBodyErrorWithGradAndHessBrent
    
    private static final int KERNEL_daffineError = 27;
    private static final int KERNEL_daffineErrorWithGradAndHess = 28;
    private static final int KERNEL_daffineErrorWithGradAndHessBrent = 32;
    private static final int KERNEL_daffineTransformImageWithBsplineInterpolation = 33;
    private static final int KERNEL_daffineSumInLocalMemoryCombined = 30;
    
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
        
        private int doubleblockSizesFPT;
        private int doubleblockSizesParallel;
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
        private CLProgram uniformBSplineTransformProgramDouble;
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
             if(uniformBSplineTransformProgramDouble != null)
			{
				uniformBSplineTransformProgramDouble.release();
				uniformBSplineTransformProgramDouble = null;
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
            if(parallelSumReductionBuffers != null)
            {
	        	for(int l = 0;l < parallelSumReductionBuffers.length;l++)
	            {
	        		if(parallelSumReductionBuffers[l] != null)
	        		{
	        			parallelSumReductionBuffers[l].release();
	        			parallelSumReductionBuffers[l] = null;
	        		}
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
            if(doubleEntryImageBuffer != null)
            {
            	doubleEntryImageBuffer.release();
            	doubleEntryImageBuffer = null;
            }
            if(doubleFullSizedGPUResidentHelperBuffer != null)
			{
				doubleFullSizedGPUResidentHelperBuffer.release();
				doubleFullSizedGPUResidentHelperBuffer = null;
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
                    //maskBuffer.getCLSize();
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
            else
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
            if(!usesFloat)
            {
                sourceDoubleSlice = new OCLSourcePyramidSlice();
                targetDoubleSlice = new OCLTargetPyramidSlice();
                // Everything can be done on the GPU
                sourceDoubleSlice.width = width;
                sourceDoubleSlice.height = height;
                sourceDoubleSlice.Image = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                //sourceDoubleSlice.Image.getCLSize();
                sourceDoubleSlice.xGradient = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                //sourceDoubleSlice.xGradient.getCLSize();
                sourceDoubleSlice.yGradient = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                //sourceDoubleSlice.yGradient.getCLSize();
                targetDoubleSlice.width = width;
                targetDoubleSlice.height = height;
                targetDoubleSlice.Coefficient = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                //targetDoubleSlice.Coefficient.getCLSize();
                // these are the conversion intermediate buffers
                doubleEntryImageBuffer = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                //doubleEntryImageBuffer.getCLSize();
                doubleFullSizedGPUResidentHelperBuffer = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                //doubleFullSizedGPUResidentHelperBuffer.getCLSize();
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
                //ourcePyramid[j].Image.getCLSize();
                sourcePyramid[j].xGradient = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                //sourcePyramid[j].xGradient.getCLSize();
                sourcePyramid[j].yGradient= context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                //sourcePyramid[j].yGradient.getCLSize();
                targetPyramid[j].Coefficient = context.createFloatBuffer((int) (width*height), GPURESIDENTRW);
                //targetPyramid[j].Coefficient.getCLSize();
                width /= 2;
                height /= 2;
            }

            if(!usesFloat)
            {
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
                //conversionEntryBuffer.getCLSize();
            }
            else
            {
                // this means that the double GPU implementation is not available so this will stay CPU side
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
            //entryImageBuffer.getCLSize();
            fullSizedGPUResidentHelperBuffer = context.createFloatBuffer((int) (sharedContext.img.dimension(0)*sharedContext.img.dimension(1)), GPURESIDENTRW);
            //fullSizedGPUResidentHelperBuffer.getCLSize();
            secondaryGPUResidentHelperBuffer = context.createFloatBuffer(((((int)sharedContext.img.dimension(0)))*(((int)sharedContext.img.dimension(1)))), GPURESIDENTRW);
            //secondaryGPUResidentHelperBuffer.getCLSize();
        }
        
        private void CompileAndSetupOpenCLKernerls() throws IOException
        {
            uniformBSplineTransformProgramKernels = new CLKernel[NR_OF_OPENCL_KERNELS];
            optimalMultiples = new long[NR_OF_OPENCL_KERNELS];
            if(usesFloat)
            {
                // only get the float part
                //uniformBSplineTransformProgram = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/HybridPrecisionBSplineTransform.cl")).build(device);
                switch(sharedContext.transformationType) {
                case TRANSLATION:
                	uniformBSplineTransformProgram = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/UniformBSplineTransform.cl")).build("-D HYBRID -D TRANSLATION", device);
                	break;
                case RIGIDBODY:
                	uniformBSplineTransformProgram = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/UniformBSplineTransform.cl")).build("-D HYBRID -D RIGIDBODY", device);
                	break;
                case SCALEDROTATION:
                	uniformBSplineTransformProgram = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/UniformBSplineTransform.cl")).build("-D HYBRID -D SCALEDROTATION", device);
					break;
                case AFFINE:
                	uniformBSplineTransformProgram = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/UniformBSplineTransform.cl")).build("-D HYBRID -D AFFINE", device);
                	break;
                }
            }
            else
            {
                // also compile the double part
            	//uniformBSplineTransformProgramDouble = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/HybridPrecisionBSplineTransform.cl")).build("-D USE_DOUBLE",device);
            	switch(sharedContext.transformationType) {
                case TRANSLATION:
                	uniformBSplineTransformProgramDouble = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/UniformBSplineTransform.cl")).build("-D HYBRID -D TRANSLATION -D USE_DOUBLE", device);
                	uniformBSplineTransformProgram = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/UniformBSplineTransform.cl")).build("-D HYBRID -D TRANSLATION", device);
                	break;
                case RIGIDBODY:
                	uniformBSplineTransformProgramDouble = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/UniformBSplineTransform.cl")).build("-D HYBRID -D RIGIDBODY -D USE_DOUBLE", device);
                	uniformBSplineTransformProgram = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/UniformBSplineTransform.cl")).build("-D HYBRID -D RIGIDBODY", device);
                	break;
                case SCALEDROTATION:
                	uniformBSplineTransformProgramDouble = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/UniformBSplineTransform.cl")).build("-D HYBRID -D SCALEDROTATION -D USE_DOUBLE", device);
                	uniformBSplineTransformProgram = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/UniformBSplineTransform.cl")).build("-D HYBRID -D SCALEDROTATION", device);
					break;
                case AFFINE:
                	uniformBSplineTransformProgramDouble = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/UniformBSplineTransform.cl")).build("-D HYBRID -D AFFINE -D USE_DOUBLE", device);
                	uniformBSplineTransformProgram = context.createProgram(getClass().getResourceAsStream("/ch/unibas/biozentrum/imagejplugins/opencl/UniformBSplineTransform.cl")).build("-D HYBRID -D AFFINE", device);
                	break;
                }
                uniformBSplineTransformProgramKernels[KERNEL_ConvertDoubleToFloat] = uniformBSplineTransformProgramDouble.createCLKernel("ConvertDoubleToFloat");
                uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2Dpremulhp] = uniformBSplineTransformProgramDouble.createCLKernel("CubicBSplinePrefilter2Dpremulhp");
                uniformBSplineTransformProgramKernels[KERNEL_dTargetedCubicBSplinePrefilter2Dpremulhp] = uniformBSplineTransformProgramDouble.createCLKernel("TargetedCubicBSplinePrefilter2Dpremulhp");
                uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DXhp] = uniformBSplineTransformProgramDouble.createCLKernel("CubicBSplinePrefilter2DXhp");
                uniformBSplineTransformProgramKernels[KERNEL_dCubicBSplinePrefilter2DYhp] = uniformBSplineTransformProgramDouble.createCLKernel("CubicBSplinePrefilter2DYhp");
                uniformBSplineTransformProgramKernels[KERNEL_dantiSymmetricFirMirrorOffBounds1DX] = uniformBSplineTransformProgramDouble.createCLKernel("antiSymmetricFirMirrorOffBounds1DX");
                uniformBSplineTransformProgramKernels[KERNEL_dantiSymmetricFirMirrorOffBounds1DY] = uniformBSplineTransformProgramDouble.createCLKernel("antiSymmetricFirMirrorOffBounds1DY");
                uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemory] = uniformBSplineTransformProgramDouble.createCLKernel("sumInLocalMemory");
                uniformBSplineTransformProgramKernels[KERNEL_dparallelGroupedSumReduction] = uniformBSplineTransformProgramDouble.createCLKernel("parallelGroupedSumReduction");
                
                switch(sharedContext.transformationType) {
                case TRANSLATION:
                    uniformBSplineTransformProgramKernels[KERNEL_dtranslationError] = uniformBSplineTransformProgramDouble.createCLKernel("translationError");
                    uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHess] = uniformBSplineTransformProgramDouble.createCLKernel("translationErrorWithGradAndHess");
                    uniformBSplineTransformProgramKernels[KERNEL_dtranslationSumInLocalMemoryCombined] = uniformBSplineTransformProgramDouble.createCLKernel("translationSumInLocalMemoryCombined");
                    uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHessBrent] = uniformBSplineTransformProgramDouble.createCLKernel("translationErrorWithGradAndHessBrent");
                    uniformBSplineTransformProgramKernels[KERNEL_dtranslationtransformImageWithBsplineInterpolation] = uniformBSplineTransformProgramDouble.createCLKernel("translationtransformImageWithBsplineInterpolation");
                    uniformBSplineTransformProgramKernels[KERNEL_dresizingTransformImageWithBsplineInterpolation] = uniformBSplineTransformProgramDouble.createCLKernel("resizingTranslationTransformImageWithBsplineInterpolation");
                    break;
                case RIGIDBODY:
                    uniformBSplineTransformProgramKernels[KERNEL_drigidBodyError] = uniformBSplineTransformProgramDouble.createCLKernel("rigidBodyError");
                    uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHess] = uniformBSplineTransformProgramDouble.createCLKernel("rigidBodyErrorWithGradAndHess");
                    uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemoryCombined] = uniformBSplineTransformProgramDouble.createCLKernel("sumInLocalMemoryCombined");
                    uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHessBrent] = uniformBSplineTransformProgramDouble.createCLKernel("rigidBodyErrorWithGradAndHessBrent");
                    uniformBSplineTransformProgramKernels[KERNEL_dtransformImageWithBsplineInterpolation] = uniformBSplineTransformProgramDouble.createCLKernel("transformImageWithBsplineInterpolation");
                    uniformBSplineTransformProgramKernels[KERNEL_dresizingTransformImageWithBsplineInterpolation] = uniformBSplineTransformProgramDouble.createCLKernel("resizingTransformImageWithBsplineInterpolation");
                    break;
                case SCALEDROTATION:
                	uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationError] = uniformBSplineTransformProgramDouble.createCLKernel("scaledRotationError");
                    uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationErrorWithGradAndHess] = uniformBSplineTransformProgramDouble.createCLKernel("scaledRotationErrorWithGradAndHess");
                    uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationSumInLocalMemoryCombined] = uniformBSplineTransformProgramDouble.createCLKernel("sumInLocalMemoryScaledRotationCombined");
                    uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationErrorWithGradAndHessBrent] = uniformBSplineTransformProgramDouble.createCLKernel("scaledRotationErrorWithGradAndHessBrent");
                    uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationTransformImageWithBsplineInterpolation] = uniformBSplineTransformProgramDouble.createCLKernel("scaledRotationTransformImageWithBsplineInterpolation");                    
                    uniformBSplineTransformProgramKernels[KERNEL_dresizingTransformImageWithBsplineInterpolation] = uniformBSplineTransformProgramDouble.createCLKernel("resizingScaledRotationTransformImageWithBsplineInterpolation");
                    break;
                case AFFINE:
                	uniformBSplineTransformProgramKernels[KERNEL_daffineError] = uniformBSplineTransformProgramDouble.createCLKernel("affineError");
                    uniformBSplineTransformProgramKernels[KERNEL_daffineErrorWithGradAndHess] = uniformBSplineTransformProgramDouble.createCLKernel("affineErrorWithGradAndHess");
                    uniformBSplineTransformProgramKernels[KERNEL_daffineSumInLocalMemoryCombined] = uniformBSplineTransformProgramDouble.createCLKernel("sumInLocalMemoryAffineCombined");
                    uniformBSplineTransformProgramKernels[KERNEL_daffineErrorWithGradAndHessBrent] = uniformBSplineTransformProgramDouble.createCLKernel("affineErrorWithGradAndHessBrent");
                    uniformBSplineTransformProgramKernels[KERNEL_daffineTransformImageWithBsplineInterpolation] = uniformBSplineTransformProgramDouble.createCLKernel("affineTransformImageWithBsplineInterpolation");
                    uniformBSplineTransformProgramKernels[KERNEL_dresizingTransformImageWithBsplineInterpolation] = uniformBSplineTransformProgramDouble.createCLKernel("resizingAffineTransformImageWithBsplineInterpolation");
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
                optimalMultiples[KERNEL_dresizingTransformImageWithBsplineInterpolation] = uniformBSplineTransformProgramKernels[KERNEL_dresizingTransformImageWithBsplineInterpolation].getPreferredWorkGroupSizeMultiple(device);

                switch(sharedContext.transformationType) {
                case TRANSLATION:
                	optimalMultiples[KERNEL_dtranslationError] = uniformBSplineTransformProgramKernels[KERNEL_dtranslationError].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_dtranslationErrorWithGradAndHess] = uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHess].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_dtranslationSumInLocalMemoryCombined] = uniformBSplineTransformProgramKernels[KERNEL_dtranslationSumInLocalMemoryCombined].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_dtranslationErrorWithGradAndHessBrent] = uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHessBrent].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_dtranslationtransformImageWithBsplineInterpolation] = uniformBSplineTransformProgramKernels[KERNEL_dtranslationtransformImageWithBsplineInterpolation].getPreferredWorkGroupSizeMultiple(device);
                    
                    doubleblockSizesParallel = (int)Math.min(Math.min(uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHessBrent].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHessBrent].getLocalMemorySize(device))/((8)*11/*11 buffers are needed*/)), blocksizeMultiplier*optimalMultiples[KERNEL_dtranslationErrorWithGradAndHessBrent]);
                    doubleblockSizesParallel -= (doubleblockSizesParallel % optimalMultiples[KERNEL_dtranslationErrorWithGradAndHessBrent]);
                    if(doubleblockSizesParallel == 0)
                    {
                        doubleblockSizesParallel = 1;// Fallback solution minimum is 1
                    }
                    
                    doublemaximumElementsForLocalFPTcombinedSum = (int) Math.min(uniformBSplineTransformProgramKernels[KERNEL_dtranslationSumInLocalMemoryCombined].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_dtranslationSumInLocalMemoryCombined].getLocalMemorySize(device))/(8));
                    break;
                case RIGIDBODY:
                	optimalMultiples[KERNEL_drigidBodyError] = uniformBSplineTransformProgramKernels[KERNEL_drigidBodyError].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_drigidBodyErrorWithGradAndHess] = uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHess].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_dsumInLocalMemoryCombined] = uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemoryCombined].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_drigidBodyErrorWithGradAndHessBrent] = uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHessBrent].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_dtransformImageWithBsplineInterpolation] = uniformBSplineTransformProgramKernels[KERNEL_dtransformImageWithBsplineInterpolation].getPreferredWorkGroupSizeMultiple(device);
                	
                    doubleblockSizesParallel = (int)Math.min(Math.min(uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHessBrent].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHessBrent].getLocalMemorySize(device))/((8)*11/*11 buffers are needed*/)), blocksizeMultiplier*optimalMultiples[KERNEL_drigidBodyErrorWithGradAndHessBrent]);
                    doubleblockSizesParallel -= (doubleblockSizesParallel % optimalMultiples[KERNEL_drigidBodyErrorWithGradAndHessBrent]);
                    if(doubleblockSizesParallel == 0)
                    {
                        doubleblockSizesParallel = 1;// Fallback solution minimum is 1
                    }
                    
                    doublemaximumElementsForLocalFPTcombinedSum = (int) Math.min(uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemoryCombined].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_dsumInLocalMemoryCombined].getLocalMemorySize(device))/(8));
                    break;
                case SCALEDROTATION:
                	optimalMultiples[KERNEL_dscaledRotationError] = uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationError].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_dscaledRotationErrorWithGradAndHess] = uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationErrorWithGradAndHess].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_dscaledRotationSumInLocalMemoryCombined] = uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationSumInLocalMemoryCombined].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_dscaledRotationErrorWithGradAndHessBrent] = uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationErrorWithGradAndHessBrent].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_dscaledRotationTransformImageWithBsplineInterpolation] = uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationTransformImageWithBsplineInterpolation].getPreferredWorkGroupSizeMultiple(device);
                    
                    doubleblockSizesParallel = (int)Math.min(Math.min(uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationErrorWithGradAndHessBrent].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationErrorWithGradAndHessBrent].getLocalMemorySize(device))/((8)*16/*16 buffers are needed*/)), blocksizeMultiplier*optimalMultiples[KERNEL_dscaledRotationErrorWithGradAndHessBrent]);
                    doubleblockSizesParallel -= (doubleblockSizesParallel % optimalMultiples[KERNEL_dscaledRotationErrorWithGradAndHessBrent]);
                    if(doubleblockSizesParallel == 0)
                    {
                        doubleblockSizesParallel = 1;// Fallback solution minimum is 1
                    }
                    
                    doublemaximumElementsForLocalFPTcombinedSum = (int) Math.min(uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationSumInLocalMemoryCombined].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationSumInLocalMemoryCombined].getLocalMemorySize(device))/(8));
                    break;
                case AFFINE:
                	optimalMultiples[KERNEL_daffineError] = uniformBSplineTransformProgramKernels[KERNEL_daffineError].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_daffineErrorWithGradAndHess] = uniformBSplineTransformProgramKernels[KERNEL_daffineErrorWithGradAndHess].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_daffineSumInLocalMemoryCombined] = uniformBSplineTransformProgramKernels[KERNEL_daffineSumInLocalMemoryCombined].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_daffineErrorWithGradAndHessBrent] = uniformBSplineTransformProgramKernels[KERNEL_daffineErrorWithGradAndHessBrent].getPreferredWorkGroupSizeMultiple(device);
                	optimalMultiples[KERNEL_daffineTransformImageWithBsplineInterpolation] = uniformBSplineTransformProgramKernels[KERNEL_daffineTransformImageWithBsplineInterpolation].getPreferredWorkGroupSizeMultiple(device);
                    
                    doubleblockSizesParallel = (int)Math.min(Math.min(uniformBSplineTransformProgramKernels[KERNEL_daffineErrorWithGradAndHessBrent].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_daffineErrorWithGradAndHessBrent].getLocalMemorySize(device))/((8)*29/*29 buffers are needed*/)), blocksizeMultiplier*optimalMultiples[KERNEL_daffineErrorWithGradAndHessBrent]);
                    doubleblockSizesParallel -= (doubleblockSizesParallel % optimalMultiples[KERNEL_daffineErrorWithGradAndHessBrent]);
                    if(doubleblockSizesParallel == 0)
                    {
                        doubleblockSizesParallel = 1;// Fallback solution minimum is 1
                    }
                    
                    doublemaximumElementsForLocalFPTcombinedSum = (int) Math.min(uniformBSplineTransformProgramKernels[KERNEL_daffineSumInLocalMemoryCombined].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_daffineSumInLocalMemoryCombined].getLocalMemorySize(device))/(8));
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
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DXhp] = uniformBSplineTransformProgram.createCLKernel("BasicToCardinal2DXhp");
            uniformBSplineTransformProgramKernels[KERNEL_BasicToCardinal2DYhp] = uniformBSplineTransformProgram.createCLKernel("BasicToCardinal2DYhp");
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DDeg7premulhp] = uniformBSplineTransformProgram.createCLKernel("CubicBSplinePrefilter2DDeg7premulhp");
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DXDeg7lp] = uniformBSplineTransformProgram.createCLKernel("fCubicBSplinePrefilter2DXDeg7lp");
            uniformBSplineTransformProgramKernels[KERNEL_CubicBSplinePrefilter2DYDeg7lp] = uniformBSplineTransformProgram.createCLKernel("fCubicBSplinePrefilter2DYDeg7lp");
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
                uniformBSplineTransformProgramKernels[KERNEL_ftranslationError] = uniformBSplineTransformProgram.createCLKernel("translationError");
                uniformBSplineTransformProgramKernels[KERNEL_ftranslationErrorWithGradAndHess] = uniformBSplineTransformProgram.createCLKernel("translationErrorWithGradAndHess");
                uniformBSplineTransformProgramKernels[KERNEL_ftranslationSumInLocalMemoryCombined] = uniformBSplineTransformProgram.createCLKernel("translationSumInLocalMemoryCombined");
                uniformBSplineTransformProgramKernels[KERNEL_ftranslationErrorWithGradAndHessBrent] = uniformBSplineTransformProgram.createCLKernel("translationErrorWithGradAndHessBrent");
                break;
            case RIGIDBODY:
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyError] = uniformBSplineTransformProgram.createCLKernel("rigidBodyError");
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHess] = uniformBSplineTransformProgram.createCLKernel("rigidBodyErrorWithGradAndHess");
                uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemoryCombined] = uniformBSplineTransformProgram.createCLKernel("sumInLocalMemoryCombined");
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent] = uniformBSplineTransformProgram.createCLKernel("rigidBodyErrorWithGradAndHessBrent");
                break;
            case SCALEDROTATION:
            	uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationError] = uniformBSplineTransformProgram.createCLKernel("scaledRotationError");
                uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationErrorWithGradAndHess] = uniformBSplineTransformProgram.createCLKernel("scaledRotationErrorWithGradAndHess");
                uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationSumInLocalMemoryCombined] = uniformBSplineTransformProgram.createCLKernel("sumInLocalMemoryScaledRotationCombined");
                uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationErrorWithGradAndHessBrent] = uniformBSplineTransformProgram.createCLKernel("scaledRotationErrorWithGradAndHessBrent");
                break;
            case AFFINE:
            	uniformBSplineTransformProgramKernels[KERNEL_faffineError] = uniformBSplineTransformProgram.createCLKernel("affineError");
                uniformBSplineTransformProgramKernels[KERNEL_faffineErrorWithGradAndHess] = uniformBSplineTransformProgram.createCLKernel("affineErrorWithGradAndHess");
                uniformBSplineTransformProgramKernels[KERNEL_faffineSumInLocalMemoryCombined] = uniformBSplineTransformProgram.createCLKernel("sumInLocalMemoryAffineCombined");
                uniformBSplineTransformProgramKernels[KERNEL_faffineErrorWithGradAndHessBrent] = uniformBSplineTransformProgram.createCLKernel("affineErrorWithGradAndHessBrent");
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
                
                blockSizesParallel = (int)Math.min(Math.min(uniformBSplineTransformProgramKernels[KERNEL_ftranslationErrorWithGradAndHessBrent].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_ftranslationErrorWithGradAndHessBrent].getLocalMemorySize(device))/((4)*7/*we need 7 buffers*/)), blocksizeMultiplier*optimalMultiples[KERNEL_ftranslationErrorWithGradAndHessBrent]);
                blockSizesParallel -= (blockSizesParallel % optimalMultiples[KERNEL_ftranslationErrorWithGradAndHessBrent]);
                if(blockSizesParallel == 0)
                {
                    blockSizesParallel = 1;// Fallback solution minimum is 1
                }
                
                maximumElementsForLocalFPTcombinedSum = (int) Math.min(uniformBSplineTransformProgramKernels[KERNEL_ftranslationSumInLocalMemoryCombined].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_ftranslationSumInLocalMemoryCombined].getLocalMemorySize(device))/(4));
                break;
            case RIGIDBODY:
            	optimalMultiples[KERNEL_rigidBodyError] = uniformBSplineTransformProgramKernels[KERNEL_rigidBodyError].getPreferredWorkGroupSizeMultiple(device);
            	optimalMultiples[KERNEL_rigidBodyErrorWithGradAndHess] = uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHess].getPreferredWorkGroupSizeMultiple(device);
            	optimalMultiples[KERNEL_sumInLocalMemoryCombined] = uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemoryCombined].getPreferredWorkGroupSizeMultiple(device);
            	optimalMultiples[KERNEL_rigidBodyErrorWithGradAndHessBrent] = uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent].getPreferredWorkGroupSizeMultiple(device);
                
                blockSizesParallel = (int)Math.min(Math.min(uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent].getLocalMemorySize(device))/((4)*11/*we need 11 buffers*/)), blocksizeMultiplier*optimalMultiples[KERNEL_rigidBodyErrorWithGradAndHessBrent]);
                blockSizesParallel -= (blockSizesParallel % optimalMultiples[KERNEL_rigidBodyErrorWithGradAndHessBrent]);
                if(blockSizesParallel == 0)
                {
                    blockSizesParallel = 1;// Fallback solution minimum is 1
                }
                
                maximumElementsForLocalFPTcombinedSum = (int) Math.min(uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemoryCombined].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_sumInLocalMemoryCombined].getLocalMemorySize(device))/(4));
                break;
            case SCALEDROTATION:
            	optimalMultiples[KERNEL_fscaledRotationError] = uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationError].getPreferredWorkGroupSizeMultiple(device);
            	optimalMultiples[KERNEL_fscaledRotationErrorWithGradAndHess] = uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationErrorWithGradAndHess].getPreferredWorkGroupSizeMultiple(device);
            	optimalMultiples[KERNEL_fscaledRotationSumInLocalMemoryCombined] = uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationSumInLocalMemoryCombined].getPreferredWorkGroupSizeMultiple(device);
            	optimalMultiples[KERNEL_fscaledRotationErrorWithGradAndHessBrent] = uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationErrorWithGradAndHessBrent].getPreferredWorkGroupSizeMultiple(device);
                
                blockSizesParallel = (int)Math.min(Math.min(uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationErrorWithGradAndHessBrent].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationErrorWithGradAndHessBrent].getLocalMemorySize(device))/((4)*16/*we need 16 buffers*/)), blocksizeMultiplier*optimalMultiples[KERNEL_fscaledRotationErrorWithGradAndHessBrent]);
                blockSizesParallel -= (blockSizesParallel % optimalMultiples[KERNEL_fscaledRotationErrorWithGradAndHessBrent]);
                if(blockSizesParallel == 0)
                {
                    blockSizesParallel = 1;// Fallback solution minimum is 1
                }
                
                maximumElementsForLocalFPTcombinedSum = (int) Math.min(uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationSumInLocalMemoryCombined].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationSumInLocalMemoryCombined].getLocalMemorySize(device))/(4));
                break;
            case AFFINE:
            	optimalMultiples[KERNEL_faffineError] = uniformBSplineTransformProgramKernels[KERNEL_faffineError].getPreferredWorkGroupSizeMultiple(device);
            	optimalMultiples[KERNEL_faffineErrorWithGradAndHess] = uniformBSplineTransformProgramKernels[KERNEL_faffineErrorWithGradAndHess].getPreferredWorkGroupSizeMultiple(device);
            	optimalMultiples[KERNEL_faffineSumInLocalMemoryCombined] = uniformBSplineTransformProgramKernels[KERNEL_faffineSumInLocalMemoryCombined].getPreferredWorkGroupSizeMultiple(device);
            	optimalMultiples[KERNEL_faffineErrorWithGradAndHessBrent] = uniformBSplineTransformProgramKernels[KERNEL_faffineErrorWithGradAndHessBrent].getPreferredWorkGroupSizeMultiple(device);
                
                blockSizesParallel = (int)Math.min(Math.min(uniformBSplineTransformProgramKernels[KERNEL_faffineErrorWithGradAndHessBrent].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_faffineErrorWithGradAndHessBrent].getLocalMemorySize(device))/((4)*29/*we need 29 buffers*/)), blocksizeMultiplier*optimalMultiples[KERNEL_faffineErrorWithGradAndHessBrent]);
                blockSizesParallel -= (blockSizesParallel % optimalMultiples[KERNEL_faffineErrorWithGradAndHessBrent]);
                if(blockSizesParallel == 0)
                {
                    blockSizesParallel = 1;// Fallback solution minimum is 1
                }
                
                maximumElementsForLocalFPTcombinedSum = (int) Math.min(uniformBSplineTransformProgramKernels[KERNEL_faffineSumInLocalMemoryCombined].getWorkGroupSize(device), (device.getLocalMemSize() - uniformBSplineTransformProgramKernels[KERNEL_faffineSumInLocalMemoryCombined].getLocalMemorySize(device))/(4));
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
        	for(int l = 0;l < parallelSumReductionBuffers.length;l++)
            {
                parallelSumReductionBuffers[l].release();
                parallelSumReductionBuffers[l] = null;
            }
        	parallelSumReductionBuffers = null;
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
            if(!usesFloat)
            {
                sourceDoubleSlice.Image.release();
                sourceDoubleSlice.Image = null;
                sourceDoubleSlice.xGradient.release();
                sourceDoubleSlice.xGradient = null;
                sourceDoubleSlice.yGradient.release();
                sourceDoubleSlice.yGradient = null;
                targetDoubleSlice.Coefficient.release();
                targetDoubleSlice.Coefficient = null;
                // these are the conversion intermediate buffers
                // no need to resize this buffer doubleEntryImageBuffer = context.createDoubleBuffer((int) (width*height), GPURESIDENTRW);
                doubleFullSizedGPUResidentHelperBuffer.release();
                doubleFullSizedGPUResidentHelperBuffer = null;
                doubleFullSizedGPUResidentHelperBuffer = context.createDoubleBuffer((int) (sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)), GPURESIDENTRW);
            }
            else
            {
                // Has to be done by the CPU so allocate the necessary buffers
                sourceImageDoubleSlice = null;
                sourcexGradientDoubleSlice = null;
                sourceyGradientDoubleSlice = null;
                //targetCoefficientDoubleSlice = new double[width*height]; no need to change the size
                CPUentryImageBuffer = new double[(int) (sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1))];
            }
            

            if(!usesFloat)
            {
            	if((sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)) > (sharedContext.img.dimension(0)*sharedContext.img.dimension(1)))
            	{
            		conversionEntryBuffer.release();
            		
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
	                else if((sharedContext.resizedTargetImage.firstElement() instanceof FloatType))
	                {
	                    conversionEntryBuffer = context.createFloatBuffer((int) (sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)), GPURESIDENTRW);
	                }
	                else
	                {
	                    // long and double are not allowed for this method
	                    throw new RuntimeException("Illegal image type");
	                }
            	}
            }

            entryImageBuffer.release();
            entryImageBuffer = null;
            fullSizedGPUResidentHelperBuffer.release();
            fullSizedGPUResidentHelperBuffer = null;
            secondaryGPUResidentHelperBuffer.release();
            secondaryGPUResidentHelperBuffer = null;
            
            //TODO: shorten
            if(!usesFloat)
            {
                // only useful if the GPU is capable of accepting double
                // Ugly code but I didn't figure out how to do it more elegantly
                if(sharedContext.img.firstElement() instanceof ByteType)
                {
                    conversionProgramKernel.rewind();
                    conversionProgramKernel.putArg(conversionEntryBuffer).putArg(doubleEntryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                    deConversionProgramKernel.rewind();
                    deConversionProgramKernel.putArg(doubleFullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)));
                }
                if(sharedContext.img.firstElement() instanceof UnsignedByteType)
                {
                    conversionProgramKernel.rewind();
                    conversionProgramKernel.putArg(conversionEntryBuffer).putArg(doubleEntryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                    deConversionProgramKernel.rewind();
                    deConversionProgramKernel.putArg(doubleFullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)));
                }
                else if((sharedContext.img.firstElement() instanceof ShortType))
                {
                    conversionProgramKernel.rewind();
                    conversionProgramKernel.putArg(conversionEntryBuffer).putArg(doubleEntryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                    deConversionProgramKernel.rewind();
                    deConversionProgramKernel.putArg(doubleFullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)));
                }
                else if((sharedContext.img.firstElement() instanceof UnsignedShortType))
                {
                    conversionProgramKernel.rewind();
                    conversionProgramKernel.putArg(conversionEntryBuffer).putArg(doubleEntryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                    deConversionProgramKernel.rewind();
                    deConversionProgramKernel.putArg(doubleFullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)));
                }
                else if((sharedContext.img.firstElement() instanceof IntType))
                {
                    conversionProgramKernel.rewind();
                    conversionProgramKernel.putArg(conversionEntryBuffer).putArg(doubleEntryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                    deConversionProgramKernel.rewind();
                    deConversionProgramKernel.putArg(doubleFullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)));
                }
                else if((sharedContext.img.firstElement() instanceof UnsignedIntType))
                {
                    conversionProgramKernel.rewind();
                    conversionProgramKernel.putArg(conversionEntryBuffer).putArg(doubleEntryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                    deConversionProgramKernel.rewind();
                    deConversionProgramKernel.putArg(doubleFullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)));
                }
                else if((sharedContext.img.firstElement() instanceof FloatType))
                {
                    conversionProgramKernel.rewind();
                    conversionProgramKernel.putArg(conversionEntryBuffer).putArg(doubleEntryImageBuffer).putArg((int)(sharedContext.img.dimension(0)*sharedContext.img.dimension(1)));
                    deConversionProgramKernel.rewind();
                    deConversionProgramKernel.putArg(doubleFullSizedGPUResidentHelperBuffer).putArg(conversionEntryBuffer).putArg((int)(sharedContext.resizedTargetImage.dimension(0)*sharedContext.resizedTargetImage.dimension(1)));
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
                        offsetx = 0.0;
                        offsety = 0.0;
                        break;
                    case RIGIDBODY:
                        ((RigidBodyTransformation)scat.transformation).angle = angle;
                        ((RigidBodyTransformation)scat.transformation).offsetx = offsetx;
                        ((RigidBodyTransformation)scat.transformation).offsety = offsety;
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
                            offsetx = 0.0;
                            offsety = 0.0;
                            break;
                        case RIGIDBODY:
                            ((RigidBodyTransformation)scat.transformation).angle = angle;
                            ((RigidBodyTransformation)scat.transformation).offsetx = offsetx;
                            ((RigidBodyTransformation)scat.transformation).offsety = offsety;
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
            }
            // The following code requires the current position to have been reset to the 0vector
            if(sharedContext.getResizeAfterRegistration())
            {
            	/*
            	 * This means the target buffer has to be resized and we can get rid of the remaining buffers
            	 * it also depends on whether we have double GPU support or not
            	 * entryImageBuffer <- source
            	 * fullSizedGPUResidentHelperBuffer <- target
            	 * then the deconversion
            	 * fullSizedGPUResidentHelperBuffer <- source
            	 * conversionEntryBuffer <- target
            	 */
            	resizeAllocatedBuffers();
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
                if(sharedContext.getResizeAfterRegistration())
            	{
                	int targetwidth = (int)sharedContext.resizedTargetImage.dimension(0);
                	int targetheight = (int)sharedContext.resizedTargetImage.dimension(1);
	                /*
	                Now that the B-spline coefficients are in the doubleEntryImageBuffer,
	                the image needs to be transformed, rescaled back and then 
	                converted back to the original image format.
	                */
	                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dresizingTransformImageWithBsplineInterpolation].getWorkGroupSize(device);  // Local work size dimensions
	                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dresizingTransformImageWithBsplineInterpolation], targetwidth*targetheight);   // rounded up to the nearest multiple of the localWorkSize
	                uniformBSplineTransformProgramKernels[KERNEL_dresizingTransformImageWithBsplineInterpolation]
	                        .putArg(doubleEntryImageBuffer)
	                        .putArg(doubleFullSizedGPUResidentHelperBuffer)
	                        .putArg(width)
	                        .putArg(height)
	                        .putArg(width * 2)
	                        .putArg(height * 2)
	                        .putArg(targetwidth)
                            .putArg(targetheight)
	                        .putArg(((RigidBodyTransformation)scat.transformation).offsetx)
	                        .putArg(((RigidBodyTransformation)scat.transformation).offsety)
	                        .putArg(Math.cos(((RigidBodyTransformation)scat.transformation).angle))
	                        .putArg(-Math.sin(((RigidBodyTransformation)scat.transformation).angle));
	                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dresizingTransformImageWithBsplineInterpolation],0,globalWorkSize,localWorkSize);
	                uniformBSplineTransformProgramKernels[KERNEL_dresizingTransformImageWithBsplineInterpolation].rewind();
            	}
                else
                {
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
	                        .putArg(width * 2)
	                        .putArg(height * 2)
	                        .putArg(((RigidBodyTransformation)scat.transformation).offsetx)
	                        .putArg(((RigidBodyTransformation)scat.transformation).offsety)
	                        .putArg(Math.cos(((RigidBodyTransformation)scat.transformation).angle))
	                        .putArg(-Math.sin(((RigidBodyTransformation)scat.transformation).angle));
	                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dtransformImageWithBsplineInterpolation],0,globalWorkSize,localWorkSize);
	                uniformBSplineTransformProgramKernels[KERNEL_dtransformImageWithBsplineInterpolation].rewind();
                }
                // Transformed and converted back to an image at the same time
                // now convert it back to the original format
                fetchTransformedImage();
            }
            else
            {
                // CPU
            	convertToBSplineCoeffCPU(width, height);

                if(!sharedContext.getResizeAfterRegistration())
            	{
                	transformImageWithBsplineInterpolation(width, height, ((RigidBodyTransformation)scat.transformation).offsetx, ((RigidBodyTransformation)scat.transformation).offsety, ((RigidBodyTransformation)scat.transformation).angle);
            		converter.deConvertTo(CPUentryImageBuffer,scat.targetArray);
            	}
            	else
            	{
            		int targetwidth = (int)sharedContext.resizedTargetImage.dimension(0);
                	int targetheight = (int)sharedContext.resizedTargetImage.dimension(1);
                	resizingTransformImageWithBsplineInterpolation(width, height, targetwidth, targetheight, ((RigidBodyTransformation)scat.transformation).offsetx, ((RigidBodyTransformation)scat.transformation).offsety, ((RigidBodyTransformation)scat.transformation).angle);
            		converter.deConvertTo(CPUentryImageBuffer,scat.sourceArray);
            	}
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
            	if(sharedContext.getResizeAfterRegistration())
            	{
            		int targetwidth = (int)sharedContext.resizedTargetImage.dimension(0);
                	int targetheight = (int)sharedContext.resizedTargetImage.dimension(1);
	                /*
	                Now that the B-spline coefficients are in the doubleEntryImageBuffer,
	                the image needs to be transformed, rescaled back and then 
	                converted back to the original image format.
	                */
	                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dresizingTransformImageWithBsplineInterpolation].getWorkGroupSize(device);  // Local work size dimensions
	                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dresizingTransformImageWithBsplineInterpolation], targetwidth*targetheight);   // rounded up to the nearest multiple of the localWorkSize
	                uniformBSplineTransformProgramKernels[KERNEL_dresizingTransformImageWithBsplineInterpolation]
	                        .putArg(doubleEntryImageBuffer)
	                        .putArg(doubleFullSizedGPUResidentHelperBuffer)
	                        .putArg(width)
	                        .putArg(height)
	                        .putArg(width * 2)
	                        .putArg(height * 2)
                            .putArg(targetwidth)
                            .putArg(targetheight)
	                        .putArg(((TranslationTransformation)scat.transformation).offsetx)
	                        .putArg(((TranslationTransformation)scat.transformation).offsety);
	                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dresizingTransformImageWithBsplineInterpolation],0,globalWorkSize,localWorkSize);
	                uniformBSplineTransformProgramKernels[KERNEL_dresizingTransformImageWithBsplineInterpolation].rewind();
            	}
            	else
            	{
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
	                        .putArg(width * 2)
	                        .putArg(height * 2)
	                        .putArg(((TranslationTransformation)scat.transformation).offsetx)
	                        .putArg(((TranslationTransformation)scat.transformation).offsety);
	                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dtranslationtransformImageWithBsplineInterpolation],0,globalWorkSize,localWorkSize);
	                uniformBSplineTransformProgramKernels[KERNEL_dtranslationtransformImageWithBsplineInterpolation].rewind();
            	}
            	// Transformed and converted back to an image at the same time
                // now convert it back to the original format
                fetchTransformedImage();
            }
            else
            {
                // CPU
            	convertToBSplineCoeffCPU(width, height);

            	if(!sharedContext.getResizeAfterRegistration())
            	{
            		translationTransformImageWithBsplineInterpolation(width,height, ((TranslationTransformation)scat.transformation).offsetx, ((TranslationTransformation)scat.transformation).offsety);
            		converter.deConvertTo(CPUentryImageBuffer,scat.targetArray);
            	}
            	else
            	{
            		int targetwidth = (int)sharedContext.resizedTargetImage.dimension(0);
                	int targetheight = (int)sharedContext.resizedTargetImage.dimension(1);
            		resizingTranslationTransformImageWithBsplineInterpolation(width, height, targetwidth, targetheight, ((TranslationTransformation)scat.transformation).offsetx, ((TranslationTransformation)scat.transformation).offsety);
            		converter.deConvertTo(CPUentryImageBuffer,scat.sourceArray);
            	}
            }
        }
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        private void transformImageScaledRotation()
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
            	if(sharedContext.getResizeAfterRegistration())
            	{
            		int targetwidth = (int)sharedContext.resizedTargetImage.dimension(0);
                	int targetheight = (int)sharedContext.resizedTargetImage.dimension(1);
	                /*
	                Now that the B-spline coefficients are in the doubleEntryImageBuffer,
	                the image needs to be transformed, rescaled back and then 
	                converted back to the original image format.
	                */
	                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dresizingTransformImageWithBsplineInterpolation].getWorkGroupSize(device);  // Local work size dimensions
	                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dresizingTransformImageWithBsplineInterpolation], targetwidth*targetheight);   // rounded up to the nearest multiple of the localWorkSize
	                uniformBSplineTransformProgramKernels[KERNEL_dresizingTransformImageWithBsplineInterpolation]
	                        .putArg(doubleEntryImageBuffer)
	                        .putArg(doubleFullSizedGPUResidentHelperBuffer)
	                        .putArg(width)
	                        .putArg(height)
	                        .putArg(width * 2)
	                        .putArg(height * 2)
                            .putArg(targetwidth)
                            .putArg(targetheight)
                            .putArg(((ScaledRotationTransformation)scat.transformation).offsetx)
                            .putArg(((ScaledRotationTransformation)scat.transformation).offsety)
                            .putArg(Math.cos(((ScaledRotationTransformation)scat.transformation).angle))
                            .putArg(-Math.sin(((ScaledRotationTransformation)scat.transformation).angle))
                            .putArg(((ScaledRotationTransformation)scat.transformation).scale);
	                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dresizingTransformImageWithBsplineInterpolation],0,globalWorkSize,localWorkSize);
	                uniformBSplineTransformProgramKernels[KERNEL_dresizingTransformImageWithBsplineInterpolation].rewind();
            	}
            	else
            	{
	                /*
	                Now that the B-spline coefficients are in the doubleEntryImageBuffer,
	                the image needs to be transformed, rescaled back and then 
	                converted back to the original image format.
	                */
	                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationTransformImageWithBsplineInterpolation].getWorkGroupSize(device);  // Local work size dimensions
	                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dscaledRotationTransformImageWithBsplineInterpolation], width*height);   // rounded up to the nearest multiple of the localWorkSize
	                uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationTransformImageWithBsplineInterpolation]
	                        .putArg(doubleEntryImageBuffer)
	                        .putArg(doubleFullSizedGPUResidentHelperBuffer)
	                        .putArg(width)
	                        .putArg(height)
	                        .putArg(width * 2)
	                        .putArg(height * 2)
	                        .putArg(((ScaledRotationTransformation)scat.transformation).offsetx)
                            .putArg(((ScaledRotationTransformation)scat.transformation).offsety)
                            .putArg(Math.cos(((ScaledRotationTransformation)scat.transformation).angle))
                            .putArg(-Math.sin(((ScaledRotationTransformation)scat.transformation).angle))
                            .putArg(((ScaledRotationTransformation)scat.transformation).scale);
	                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationTransformImageWithBsplineInterpolation],0,globalWorkSize,localWorkSize);
	                uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationTransformImageWithBsplineInterpolation].rewind();
            	}
            	// Transformed and converted back to an image at the same time
                // now convert it back to the original format
                fetchTransformedImage();
            }
            else
            {
                // CPU
            	convertToBSplineCoeffCPU(width, height);

            	if(!sharedContext.getResizeAfterRegistration())
            	{
            		scaledRotationTransformImageWithBsplineInterpolation(width,height, ((ScaledRotationTransformation)scat.transformation).offsetx, ((ScaledRotationTransformation)scat.transformation).offsety, ((ScaledRotationTransformation)scat.transformation).angle, ((ScaledRotationTransformation)scat.transformation).scale);
            		converter.deConvertTo(CPUentryImageBuffer,scat.targetArray);
            	}
            	else
            	{
            		int targetwidth = (int)sharedContext.resizedTargetImage.dimension(0);
                	int targetheight = (int)sharedContext.resizedTargetImage.dimension(1);
            		resizingScaledRotationTransformImageWithBsplineInterpolation(width, height, targetwidth, targetheight, ((ScaledRotationTransformation)scat.transformation).offsetx, ((ScaledRotationTransformation)scat.transformation).offsety, ((ScaledRotationTransformation)scat.transformation).angle, ((ScaledRotationTransformation)scat.transformation).scale);
            		converter.deConvertTo(CPUentryImageBuffer,scat.sourceArray);
            	}
            }
        }
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        private void transformImageAffine()
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
            	if(sharedContext.getResizeAfterRegistration())
            	{
            		int targetwidth = (int)sharedContext.resizedTargetImage.dimension(0);
                	int targetheight = (int)sharedContext.resizedTargetImage.dimension(1);
	                /*
	                Now that the B-spline coefficients are in the doubleEntryImageBuffer,
	                the image needs to be transformed, rescaled back and then 
	                converted back to the original image format.
	                */
	                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_dresizingTransformImageWithBsplineInterpolation].getWorkGroupSize(device);  // Local work size dimensions
	                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dresizingTransformImageWithBsplineInterpolation], targetwidth*targetheight);   // rounded up to the nearest multiple of the localWorkSize
	                uniformBSplineTransformProgramKernels[KERNEL_dresizingTransformImageWithBsplineInterpolation]
	                        .putArg(doubleEntryImageBuffer)
	                        .putArg(doubleFullSizedGPUResidentHelperBuffer)
	                        .putArg(width)
	                        .putArg(height)
	                        .putArg(width * 2)
	                        .putArg(height * 2)
                            .putArg(targetwidth)
                            .putArg(targetheight)
                            .putArg(((AffineTransformation)scat.transformation).offsetx)
                            .putArg(((AffineTransformation)scat.transformation).offsety)
                            .putArg(((AffineTransformation)scat.transformation).a11)
                            .putArg(((AffineTransformation)scat.transformation).a12)
                            .putArg(((AffineTransformation)scat.transformation).a21)
                            .putArg(((AffineTransformation)scat.transformation).a22);
	                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dresizingTransformImageWithBsplineInterpolation],0,globalWorkSize,localWorkSize);
	                uniformBSplineTransformProgramKernels[KERNEL_dresizingTransformImageWithBsplineInterpolation].rewind();
            	}
            	else
            	{
	                /*
	                Now that the B-spline coefficients are in the doubleEntryImageBuffer,
	                the image needs to be transformed, rescaled back and then 
	                converted back to the original image format.
	                */
	                localWorkSize = (int)uniformBSplineTransformProgramKernels[KERNEL_daffineTransformImageWithBsplineInterpolation].getWorkGroupSize(device);  // Local work size dimensions
	                globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_daffineTransformImageWithBsplineInterpolation], width*height);   // rounded up to the nearest multiple of the localWorkSize
	                uniformBSplineTransformProgramKernels[KERNEL_daffineTransformImageWithBsplineInterpolation]
	                        .putArg(doubleEntryImageBuffer)
	                        .putArg(doubleFullSizedGPUResidentHelperBuffer)
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
	                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_daffineTransformImageWithBsplineInterpolation],0,globalWorkSize,localWorkSize);
	                uniformBSplineTransformProgramKernels[KERNEL_daffineTransformImageWithBsplineInterpolation].rewind();
            	}
            	// Transformed and converted back to an image at the same time
                // now convert it back to the original format
                fetchTransformedImage();
            }
            else
            {
                // CPU
            	convertToBSplineCoeffCPU(width, height);

            	if(!sharedContext.getResizeAfterRegistration())
            	{
            		affineTransformImageWithBsplineInterpolation(width,height, ((AffineTransformation)scat.transformation).offsetx, ((AffineTransformation)scat.transformation).offsety, ((AffineTransformation)scat.transformation).a11, ((AffineTransformation)scat.transformation).a12, ((AffineTransformation)scat.transformation).a21, ((AffineTransformation)scat.transformation).a22);
            		converter.deConvertTo(CPUentryImageBuffer,scat.targetArray);
            	}
            	else
            	{
            		int targetwidth = (int)sharedContext.resizedTargetImage.dimension(0);
                	int targetheight = (int)sharedContext.resizedTargetImage.dimension(1);
            		resizingAffineTransformImageWithBsplineInterpolation(width, height, targetwidth, targetheight, ((AffineTransformation)scat.transformation).offsetx, ((AffineTransformation)scat.transformation).offsety, ((AffineTransformation)scat.transformation).a11, ((AffineTransformation)scat.transformation).a12, ((AffineTransformation)scat.transformation).a21, ((AffineTransformation)scat.transformation).a22);
            		converter.deConvertTo(CPUentryImageBuffer,scat.sourceArray);
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
            double coordx;
            double coordy;
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
                    	PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                    	PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                        
                    	CPUentryImageBuffer[nIndex] = PlainJavaCPUAligner.interpolateCubicBSpline(PlainJavaCPUAligner.getFractional(coordx), PlainJavaCPUAligner.getFractional(coordy), xInterpolationIndices, yInterpolationIndices, targetCoefficientDoubleSlice);
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

        private void resizingTranslationTransformImageWithBsplineInterpolation(final int width, final int height, final int targetwidth, final int targetheight, double currentoffsetx, double currentoffsety)
        {
            /*
            Requires the coefficients to be in entryImageBuffers and the output
            will be in fullSizedHelperBuffer
            */
            int doubleWidth = width*2;
            int doubleHeight = height*2;
            int nIndex = 0;
            double coordx;
            double coordy;
            int mskx;
            int msky;
            for(int i = 0;i < targetheight;i++)
            {
                // First walk along the Y-vector direction and reset the X-position (otherwise the
                // y position is initially correct and then lagging behind by one all the time)
                coordx = currentoffsetx;
                coordy = currentoffsety + ((double)i);
                for(int n = 0;n < targetwidth;n++,nIndex++)
                {
                    mskx = (int)Math.round(coordx);
                    msky = (int)Math.round(coordy);
                    if((mskx >= 0)&&(mskx < width)&&(msky >= 0)&&(msky < height))
                    {
                    	PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                    	PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                        
                    	CPUentryImageBuffer[nIndex] = PlainJavaCPUAligner.interpolateCubicBSpline(PlainJavaCPUAligner.getFractional(coordx), PlainJavaCPUAligner.getFractional(coordy), xInterpolationIndices, yInterpolationIndices, targetCoefficientDoubleSlice);
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
        
        private void transformImageWithBsplineInterpolation(final int width, final int height, double currentoffsetx, double currentoffsety, double currentangle)
        {
            /*
            Requires the coefficients to be in entryImageBuffers and the output
            will be in fullSizedHelperBuffer
            */
            int doubleWidth = width*2;
            int doubleHeight = height*2;
            int nIndex = 0;
            double xvecx = Math.cos(currentangle);
            double xvecy = -Math.sin(currentangle);
            double yvecx = -xvecy;
            double yvecy = xvecx;
            double coordx;
            double coordy;
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
                        PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                        PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                        
                        CPUentryImageBuffer[nIndex] = PlainJavaCPUAligner.interpolateCubicBSpline(PlainJavaCPUAligner.getFractional(coordx), PlainJavaCPUAligner.getFractional(coordy), xInterpolationIndices, yInterpolationIndices, targetCoefficientDoubleSlice);
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
        
        private void resizingTransformImageWithBsplineInterpolation(final int width, final int height, final int targetwidth, final int targetheight, double currentoffsetx, double currentoffsety, double currentangle)
        {
            /*
            Requires the coefficients to be in entryImageBuffers and the output
            will be in fullSizedHelperBuffer
            */
            int doubleWidth = width*2;
            int doubleHeight = height*2;
            int nIndex = 0;
            double xvecx = Math.cos(currentangle);
            double xvecy = -Math.sin(currentangle);
            double yvecx = -xvecy;
            double yvecy = xvecx;
            double coordx;
            double coordy;
            int mskx;
            int msky;
            for(int i = 0;i < targetheight;i++)
            {
                // First walk along the Y-vector direction and reset the X-position (otherwise the
                // y position is initially correct and then lagging behind by one all the time)
                coordx = currentoffsetx + ((double)i) * yvecx;
                coordy = currentoffsety + ((double)i) * yvecy;
                for(int n = 0;n < targetwidth;n++,nIndex++)
                {
                    mskx = (int)Math.round(coordx);
                    msky = (int)Math.round(coordy);
                    if((mskx >= 0)&&(mskx < width)&&(msky >= 0)&&(msky < height))
                    {
                    	PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                    	PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                        
                    	CPUentryImageBuffer[nIndex] = PlainJavaCPUAligner.interpolateCubicBSpline(PlainJavaCPUAligner.getFractional(coordx), PlainJavaCPUAligner.getFractional(coordy), xInterpolationIndices, yInterpolationIndices, targetCoefficientDoubleSlice);
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

        private void scaledRotationTransformImageWithBsplineInterpolation(final int width, final int height, double currentoffsetx, double currentoffsety, double currentangle, double currentscale)
        {
            /*
            Requires the coefficients to be in entryImageBuffers and the output
            will be in fullSizedHelperBuffer
            */
            int doubleWidth = width*2;
            int doubleHeight = height*2;
            int nIndex = 0;
            double xvecx = Math.cos(currentangle) * currentscale;
            double xvecy = -Math.sin(currentangle) * currentscale;
            double yvecx = -xvecy;
            double yvecy = xvecx;
            double coordx;
            double coordy;
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
                    	PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                    	PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                        
                    	CPUentryImageBuffer[nIndex] = PlainJavaCPUAligner.interpolateCubicBSpline(PlainJavaCPUAligner.getFractional(coordx), PlainJavaCPUAligner.getFractional(coordy), xInterpolationIndices, yInterpolationIndices, targetCoefficientDoubleSlice);
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
        
        private void resizingScaledRotationTransformImageWithBsplineInterpolation(final int width, final int height, final int targetwidth, final int targetheight, double currentoffsetx, double currentoffsety, double currentangle, double currentscale)
        {
            /*
            Requires the coefficients to be in entryImageBuffers and the output
            will be in fullSizedHelperBuffer
            */
            int doubleWidth = width*2;
            int doubleHeight = height*2;
            int nIndex = 0;
            double xvecx = Math.cos(currentangle) * currentscale;
            double xvecy = -Math.sin(currentangle) * currentscale;
            double yvecx = -xvecy;
            double yvecy = xvecx;
            double coordx;
            double coordy;
            int mskx;
            int msky;
            for(int i = 0;i < targetheight;i++)
            {
                // First walk along the Y-vector direction and reset the X-position (otherwise the
                // y position is initially correct and then lagging behind by one all the time)
                coordx = currentoffsetx + ((double)i) * yvecx;
                coordy = currentoffsety + ((double)i) * yvecy;
                for(int n = 0;n < targetwidth;n++,nIndex++)
                {
                    mskx = (int)Math.round(coordx);
                    msky = (int)Math.round(coordy);
                    if((mskx >= 0)&&(mskx < width)&&(msky >= 0)&&(msky < height))
                    {
                    	PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                    	PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                        
                    	CPUentryImageBuffer[nIndex] = PlainJavaCPUAligner.interpolateCubicBSpline(PlainJavaCPUAligner.getFractional(coordx), PlainJavaCPUAligner.getFractional(coordy), xInterpolationIndices, yInterpolationIndices, targetCoefficientDoubleSlice);
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

        private void affineTransformImageWithBsplineInterpolation(final int width, final int height, double currentoffsetx, double currentoffsety, double currenta11, double currenta12, double currenta21, double currenta22)
        {
            /*
            Requires the coefficients to be in entryImageBuffers and the output
            will be in fullSizedHelperBuffer
            */
            int doubleWidth = width*2;
            int doubleHeight = height*2;
            int nIndex = 0;
            double xvecx = currenta11;
            double xvecy = currenta21;
            double yvecx = currenta12;
            double yvecy = currenta22;
            double coordx;
            double coordy;
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
                    	PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                    	PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                        
                    	CPUentryImageBuffer[nIndex] = PlainJavaCPUAligner.interpolateCubicBSpline(PlainJavaCPUAligner.getFractional(coordx), PlainJavaCPUAligner.getFractional(coordy), xInterpolationIndices, yInterpolationIndices, targetCoefficientDoubleSlice);
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

        private void resizingAffineTransformImageWithBsplineInterpolation(final int width, final int height, final int targetwidth, final int targetheight, double currentoffsetx, double currentoffsety, double currenta11, double currenta12, double currenta21, double currenta22)
        {
            /*
            Requires the coefficients to be in entryImageBuffers and the output
            will be in fullSizedHelperBuffer
            */
            int doubleWidth = width*2;
            int doubleHeight = height*2;
            int nIndex = 0;
            double xvecx = currenta11;
            double xvecy = currenta21;
            double yvecx = currenta12;
            double yvecy = currenta22;
            double coordx;
            double coordy;
            int mskx;
            int msky;
            for(int i = 0;i < targetheight;i++)
            {
                // First walk along the Y-vector direction and reset the X-position (otherwise the
                // y position is initially correct and then lagging behind by one all the time)
                coordx = currentoffsetx + ((double)i) * yvecx;
                coordy = currentoffsety + ((double)i) * yvecy;
                for(int n = 0;n < targetwidth;n++,nIndex++)
                {
                    mskx = (int)Math.round(coordx);
                    msky = (int)Math.round(coordy);
                    if((mskx >= 0)&&(mskx < width)&&(msky >= 0)&&(msky < height))
                    {
                    	PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                    	PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                        
                    	CPUentryImageBuffer[nIndex] = PlainJavaCPUAligner.interpolateCubicBSpline(PlainJavaCPUAligner.getFractional(coordx), PlainJavaCPUAligner.getFractional(coordy), xInterpolationIndices, yInterpolationIndices, targetCoefficientDoubleSlice);
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

        private void fetchTransformedImage()
        {
            // This function is only called from the GPU code so one can assume !usesFloat == true
            // WARNING: this function overwrites the original image data without a chance to recover it!!!
        	if(!sharedContext.getResizeAfterRegistration())
        	{
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
        	else
        	{
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
            	for(int i = pyramidDepth - 1;i > 0;i--)
                {
                    iterationPower /= 2;
                    inverseMarquardtLevenbergScaledRotationOptimization(i);
                    // scale up (but the rotation is not scale dependent so simply scale up the translation)
                    offsetx *= 2.0;
                    offsety *= 2.0;
                }
                iterationPower /= 2;
                inverseMarquardtLevenbergScaledRotationOptimization(0);
                
                // With the perfect initial guess of the parameters run one last round at double precision
                doubleInverseMarquardtLevenbergScaledRotationOptimization();
                break;
            case AFFINE:
            	for(int i = pyramidDepth - 1;i > 0;i--)
                {
                    iterationPower /= 2;
                    inverseMarquardtLevenbergAffineOptimization(i);
                    // scale up (but the rotation is not scale dependent so simply scale up the translation)
                    offsetx *= 2.0;
                    offsety *= 2.0;
                }
                iterationPower /= 2;
                inverseMarquardtLevenbergAffineOptimization(0);
                
                // With the perfect initial guess of the parameters run one last round at double precision
                doubleInverseMarquardtLevenbergAffineOptimization();
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
        
        /*
         * ========================================================================================
         * inverseMarquardtLevenbergAffineOptimization
         * ========================================================================================
         *
         * Paper reference: Thévenaz et al., "A Pyramid Approach to Subpixel Registration Based
         * on Intensity," IEEE TIP 1998, Section III-A.
         *
         * This implements the modified Levenberg-Marquardt optimizer for affine registration.
         *
         * The affine transformation maps output pixel (n, i) to source coordinates:
         *   x' = a11*n + a12*i + tx
         *   y' = a21*n + a22*i + ty
         *
         * Parameter vector: p = (a11, a12, a21, a22, tx, ty), dimension = 6.
         *
         * At identity: a11=1, a12=0, a21=0, a22=1, tx=0, ty=0.
         *
         * The optimization minimizes E(p) = (1/|Ω|) Σ [f(x) - g(T_p(x))]²
         * using the "inverse" approach where the Jacobian is built from the SOURCE image
         * gradients ∇f, which remain constant across iterations. This avoids recomputing
         * the Hessian when the parameters change, as described in Section III-A of the paper.
         *
         * LM damping: The pseudoHessian is constructed by copying ONLY the diagonal of the
         * Gauss-Newton Hessian H, scaled by (1+λ):
         *   pseudoH[k][k] = (1+λ) * H[k][k]
         * with all off-diagonal entries left at zero. This reduces the damped step to:
         *   δp_k = g_k / ((1+λ) * H_kk)
         * which is an independent damped gradient descent along each parameter axis.
         * This is a deliberate simplification inherited from the original TurboReg design:
         * - Computationally cheaper: O(n) diagonal inversion instead of O(n³) full inversion.
         * - More numerically stable: avoids ill-conditioning from off-diagonal coupling.
         * - For large λ (far from optimum), LM reduces to gradient descent anyway.
         * The full Hessian (with off-diagonals) is inverted only for the FINAL undamped
         * Gauss-Newton step after the LM loop converges.
         *
         * Update signs: For affine, ALL parameters live in a vector space (unlike the SE(2)
         * group for rigid body). The update is purely additive with POSITIVE sign:
         *   a11_new = a11 + δa11
         *   a12_new = a12 + δa12
         *   ...
         *   tx_new  = tx  + δtx
         *   ty_new  = ty  + δty
         *
         * This is correct because:
         * - The gradient g_k = Σ r_i * (∂f/∂p_k) points in the direction that increases
         *   the dot product of residual with the Jacobian column.
         * - The Gauss-Newton step δp = H⁻¹ g directly gives the parameter increment
         *   that reduces the least-squares error.
         * - Unlike rigid body (where the angle update requires a MINUS sign due to the
         *   inverse/compositional convention and subsequent SE(2) group composition),
         *   affine parameters are simply linear coefficients with no such group structure.
         *
         * Contrast with rigid body:
         * - Rigid body: angle update is SUBTRACTED (currentangle = angle - update[0])
         *   because the angle parameterizes a rotation group, and the "inverse" Jacobian
         *   convention means the gradient points opposite to the forward rotation direction.
         *   The offsets are then COMPOSED through the incremental rotation matrix.
         * - Affine: all 6 parameters are ADDED directly. No group composition is needed.
         *   The affine matrix entries (a11, a12, a21, a22) are dimensionless linear
         *   coefficients, and (tx, ty) are translations — all updated additively.
         *
         * Displacement convergence criterion:
         *   displacement = sqrt(δtx² + δty²)
         *                + 0.25 * diagonal * (|δa11| + |δa12| + |δa21| + |δa22|)
         * The matrix element updates are converted to approximate pixel displacements
         * by multiplying by 0.25 × the image diagonal. This heuristic estimates the
         * maximum pixel displacement caused by a small change in a matrix coefficient
         * (analogous to the 0.25*diagonal*|δθ| term in rigid body).
         */
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

                /*
                 * Evaluate the MSE at the trial point. This also recomputes gradient
                 * and Hessian for the next iteration (the "accelerated" variant from
                 * the paper always recomputes, since the source gradients are constant
                 * but the B-spline interpolation coordinates change).
                 */
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
                	Arrays.fill(pseudoHessian[k], 0.0);
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
                	Arrays.fill(pseudoHessian[k], 0.0);
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

        private void doubleInverseMarquardtLevenbergScaledRotationOptimization()
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
            bestMeanSquares = doubleGetScaledRotationMeanSquares(offsetx,offsety,this.angle,this.scale);
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
                displacement = Math.sqrt(update[2] * update[2] + update[3] * update[3]) + 0.25 * Math.sqrt((double)(targetPyramid[0].width * targetPyramid[0].width) + (double)(targetPyramid[0].height * targetPyramid[0].height)) * (Math.abs(update[0]) + Math.abs(update[1]));
                c = Math.cos(update[1]);
                s = Math.sin(update[1]);
                currentoffsetx = ((offsetx + update[2]) * c - (offsety + update[3]) * s) * (1.0 + update[0]);
                currentoffsety = ((offsetx + update[2]) * s + (offsety + update[3]) * c) * (1.0 + update[0]);
                meanSquares = doubleGetScaledRotationMeanSquares(currentoffsetx,currentoffsety,currentangle,currentscale);

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
            meanSquares = doubleGetScaledRotationMeanSquaresWithoutHessian(currentoffsetx,currentoffsety,currentangle,currentscale);
            iteration++;
            if (meanSquares < bestMeanSquares) {
                offsetx = currentoffsetx;
                offsety = currentoffsety;
                this.angle = currentangle;
                this.scale = currentscale;
            }
        }
        
        /*
         * ========================================================================================
         * inverseMarquardtLevenbergAffineOptimization
         * ========================================================================================
         *
         * Paper reference: Thévenaz et al., "A Pyramid Approach to Subpixel Registration Based
         * on Intensity," IEEE TIP 1998, Section III-A.
         *
         * This implements the modified Levenberg-Marquardt optimizer for affine registration.
         *
         * The affine transformation maps output pixel (n, i) to source coordinates:
         *   x' = a11*n + a12*i + tx
         *   y' = a21*n + a22*i + ty
         *
         * Parameter vector: p = (a11, a12, a21, a22, tx, ty), dimension = 6.
         *
         * At identity: a11=1, a12=0, a21=0, a22=1, tx=0, ty=0.
         *
         * The optimization minimizes E(p) = (1/|Ω|) Σ [f(x) - g(T_p(x))]²
         * using the "inverse" approach where the Jacobian is built from the SOURCE image
         * gradients ∇f, which remain constant across iterations. This avoids recomputing
         * the Hessian when the parameters change, as described in Section III-A of the paper.
         *
         * LM damping: The pseudoHessian is constructed by copying ONLY the diagonal of the
         * Gauss-Newton Hessian H, scaled by (1+λ):
         *   pseudoH[k][k] = (1+λ) * H[k][k]
         * with all off-diagonal entries left at zero. This reduces the damped step to:
         *   δp_k = g_k / ((1+λ) * H_kk)
         * which is an independent damped gradient descent along each parameter axis.
         * This is a deliberate simplification inherited from the original TurboReg design:
         * - Computationally cheaper: O(n) diagonal inversion instead of O(n³) full inversion.
         * - More numerically stable: avoids ill-conditioning from off-diagonal coupling.
         * - For large λ (far from optimum), LM reduces to gradient descent anyway.
         * The full Hessian (with off-diagonals) is inverted only for the FINAL undamped
         * Gauss-Newton step after the LM loop converges.
         *
         * Update signs: For affine, ALL parameters live in a vector space (unlike the SE(2)
         * group for rigid body). The update is purely additive with POSITIVE sign:
         *   a11_new = a11 + δa11
         *   a12_new = a12 + δa12
         *   ...
         *   tx_new  = tx  + δtx
         *   ty_new  = ty  + δty
         *
         * This is correct because:
         * - The gradient g_k = Σ r_i * (∂f/∂p_k) points in the direction that increases
         *   the dot product of residual with the Jacobian column.
         * - The Gauss-Newton step δp = H⁻¹ g directly gives the parameter increment
         *   that reduces the least-squares error.
         * - Unlike rigid body (where the angle update requires a MINUS sign due to the
         *   inverse/compositional convention and subsequent SE(2) group composition),
         *   affine parameters are simply linear coefficients with no such group structure.
         *
         * Contrast with rigid body:
         * - Rigid body: angle update is SUBTRACTED (currentangle = angle - update[0])
         *   because the angle parameterizes a rotation group, and the "inverse" Jacobian
         *   convention means the gradient points opposite to the forward rotation direction.
         *   The offsets are then COMPOSED through the incremental rotation matrix.
         * - Affine: all 6 parameters are ADDED directly. No group composition is needed.
         *   The affine matrix entries (a11, a12, a21, a22) are dimensionless linear
         *   coefficients, and (tx, ty) are translations — all updated additively.
         *
         * Displacement convergence criterion:
         *   displacement = sqrt(δtx² + δty²)
         *                + 0.25 * diagonal * (|δa11| + |δa12| + |δa21| + |δa22|)
         * The matrix element updates are converted to approximate pixel displacements
         * by multiplying by 0.25 × the image diagonal. This heuristic estimates the
         * maximum pixel displacement caused by a small change in a matrix coefficient
         * (analogous to the 0.25*diagonal*|δθ| term in rigid body).
         */
        private void doubleInverseMarquardtLevenbergAffineOptimization()
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
            bestMeanSquares = doubleGetAffineMeanSquares(offsetx, offsety, a11, a12, a21, a22);
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
                        + 0.5 * (double)targetPyramid[0].width
                          * (Math.abs(update[0]) + Math.abs(update[2]))
                        + 0.5 * (double)targetPyramid[0].height
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

                /*
                 * Evaluate the MSE at the trial point. This also recomputes gradient
                 * and Hessian for the next iteration (the "accelerated" variant from
                 * the paper always recomputes, since the source gradients are constant
                 * but the B-spline interpolation coordinates change).
                 */
                meanSquares = doubleGetAffineMeanSquares(currentoffsetx, currentoffsety,
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

            meanSquares = doubleGetAffineMeanSquaresWithoutHessian(currentoffsetx, currentoffsety,
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
                        .putArg((int)targetPyramid[pyramidIndex].width * 2)
                        .putArg((int)targetPyramid[pyramidIndex].height * 2)
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
                
                int nrOfBlocks = ((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))%blockSizesParallel;
                nrOfBlocks = nrOfBlocks == 0 ? (int)Math.min(((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))/blockSizesParallel,maximumSumReductionBlockNr) : (int)Math.min((((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))+(blockSizesParallel-nrOfBlocks))/blockSizesParallel,maximumSumReductionBlockNr);
                
                int globalWorkSize = nrOfBlocks * blockSizesParallel;
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
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/);
                
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
                
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_ftranslationErrorWithGradAndHessBrent],0,globalWorkSize,blockSizesParallel);
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
                        .putArg((int)targetPyramid[pyramidIndex].height * 2)
                        .putArg((float)currentoffsetx)
                        .putArg((float)curentoffsety)
                        .putArg((float)Math.cos(currentangle))
                        .putArg((float)-Math.sin(currentangle));

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
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/);
                
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent]
                        .putArg((int)sourcePyramid[pyramidIndex].width)
                        .putArg((int)sourcePyramid[pyramidIndex].height)
                        .putArg((int)targetPyramid[pyramidIndex].width)
                        .putArg((int)targetPyramid[pyramidIndex].height);

                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent]
                        .putArg((float)currentoffsetx)
                        .putArg((float)curentoffsety)
                        .putArg((float)Math.cos(currentangle))
                        .putArg((float)-Math.sin(currentangle));
               
                uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent]
                        .putArg((int)targetPyramid[pyramidIndex].width*2)
                        .putArg((int)targetPyramid[pyramidIndex].height*2);
                
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_rigidBodyErrorWithGradAndHessBrent],0,globalWorkSize,blockSizesParallel);
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
                uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationErrorWithGradAndHess]
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
                        .putArg((int)targetPyramid[pyramidIndex].height * 2)
                        .putArg((float)currentoffsetx)
                        .putArg((float)curentoffsety)
                        .putArg((float)Math.cos(currentangle))
                        .putArg((float)-Math.sin(currentangle))
                        .putArg((float)currentscale);

                int localWorkSize = (int)Math.min(uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationErrorWithGradAndHess].getWorkGroupSize(device),blocksizeMultiplier*optimalMultiples[KERNEL_fscaledRotationErrorWithGradAndHess]);  // Local work size dimensions
                int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_fscaledRotationErrorWithGradAndHess], (int)(sourcePyramid[pyramidIndex].width*sourcePyramid[pyramidIndex].height));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationErrorWithGradAndHess],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationErrorWithGradAndHess].rewind();
                
                /*
                Because each of the following kernels is synchronized by barriers on the GPU they can only be executed on a single compute device.
                Therefore, using an async queue may increase the speed by allowing multiple kernels to execute in parallel (if local memory permits)
                */
                localWorkSize = halfReductionSize % optimalMultiples[KERNEL_fscaledRotationSumInLocalMemoryCombined] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_fscaledRotationSumInLocalMemoryCombined] - (halfReductionSize % optimalMultiples[KERNEL_fscaledRotationSumInLocalMemoryCombined])), maximumElementsForLocalFPTcombinedSum);
                globalWorkSize = localWorkSize;
                uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationSumInLocalMemoryCombined]
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
                        .putNullArg(localWorkSize*4 /*size in bytes of local mem allocation*/)
                        .putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationSumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationSumInLocalMemoryCombined].rewind();
                
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
                buffer = queue.putMapBuffer(gradient3, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                gradient[3] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(gradient3, buffer);
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
                buffer = queue.putMapBuffer(hessian03, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[0][3] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian03, buffer);
                buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[1][1] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian11, buffer);
                buffer = queue.putMapBuffer(hessian12, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[1][2] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian12, buffer);
                buffer = queue.putMapBuffer(hessian13, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[1][3] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian13, buffer);
                buffer = queue.putMapBuffer(hessian22, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[2][2] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian22, buffer);
                buffer = queue.putMapBuffer(hessian23, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[2][3] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian23, buffer);
                buffer = queue.putMapBuffer(hessian33, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[3][3] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian33, buffer);
            }
            else
            {
                
                int nrOfBlocks = ((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))%blockSizesParallel;
                nrOfBlocks = nrOfBlocks == 0 ? (int)Math.min(((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))/blockSizesParallel,maximumSumReductionBlockNr) : (int)Math.min((((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))+(blockSizesParallel-nrOfBlocks))/blockSizesParallel,maximumSumReductionBlockNr);
                
                int globalWorkSize = nrOfBlocks * blockSizesParallel;
                uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationErrorWithGradAndHessBrent]
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
                
                uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationErrorWithGradAndHessBrent]
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/);
                
                uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationErrorWithGradAndHessBrent]
                        .putArg((int)sourcePyramid[pyramidIndex].width)
                        .putArg((int)sourcePyramid[pyramidIndex].height)
                        .putArg((int)targetPyramid[pyramidIndex].width)
                        .putArg((int)targetPyramid[pyramidIndex].height);

                uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationErrorWithGradAndHessBrent]
                        .putArg((float)currentoffsetx)
                        .putArg((float)curentoffsety)
                        .putArg((float)Math.cos(currentangle))
                        .putArg((float)-Math.sin(currentangle))
                        .putArg((float)currentscale);
               
                uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationErrorWithGradAndHessBrent]
                        .putArg((int)targetPyramid[pyramidIndex].width*2)
                        .putArg((int)targetPyramid[pyramidIndex].height*2);
                
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationErrorWithGradAndHessBrent],0,globalWorkSize,blockSizesParallel);
                uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationErrorWithGradAndHessBrent].rewind();
                
                // use in-memory summations (also async) each with its own buffer to do the final reduction)
                halfReductionSize = (int) ((nrOfBlocks+(nrOfBlocks%2))/2);
                int localWorkSize = halfReductionSize % optimalMultiples[KERNEL_fscaledRotationSumInLocalMemoryCombined] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_fscaledRotationSumInLocalMemoryCombined] - (halfReductionSize % optimalMultiples[KERNEL_fscaledRotationSumInLocalMemoryCombined])), maximumElementsForLocalFPTcombinedSum);
                globalWorkSize = localWorkSize;
                uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationSumInLocalMemoryCombined]
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
                        .putNullArg(halfReductionSize*4 /*size in bytes of local mem allocation*/)
                        .putArg((int)(nrOfBlocks));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationSumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationSumInLocalMemoryCombined].rewind();
                
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
                
                buffer = queue.putMapBuffer(gradient3, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                gradient[3] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(gradient3, buffer);
                        
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
                
                buffer = queue.putMapBuffer(hessian03, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[0][3] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian03, buffer);
                        
                buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[1][1] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian11, buffer);
                        
                buffer = queue.putMapBuffer(hessian12, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[1][2] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian12, buffer);
                
                buffer = queue.putMapBuffer(hessian13, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[1][3] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian13, buffer);
                        
                buffer = queue.putMapBuffer(hessian22, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[2][2] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian22, buffer);
                
                buffer = queue.putMapBuffer(hessian23, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[2][3] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian23, buffer);
                
                buffer = queue.putMapBuffer(hessian33, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[3][3] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian33, buffer);
            }
            //symmetrize hessian
            for (int i = 1; (i < 4); i++) {
                for (int j = 0; (j < i); j++) {
                        hessian[i][j] = hessian[j][i];
                }
            }
            return mse/area;
        }
        
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
                uniformBSplineTransformProgramKernels[KERNEL_faffineErrorWithGradAndHess]
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
                        .putArg((int)targetPyramid[pyramidIndex].height * 2)
                        .putArg((float)currentoffsetx)
                        .putArg((float)curentoffsety)
                        .putArg((float)currenta11)
                        .putArg((float)currenta12)
                        .putArg((float)currenta21)
                        .putArg((float)currenta22);

                int localWorkSize = (int)Math.min(uniformBSplineTransformProgramKernels[KERNEL_faffineErrorWithGradAndHess].getWorkGroupSize(device),blocksizeMultiplier*optimalMultiples[KERNEL_faffineErrorWithGradAndHess]);  // Local work size dimensions
                int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_faffineErrorWithGradAndHess], (int)(sourcePyramid[pyramidIndex].width*sourcePyramid[pyramidIndex].height));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_faffineErrorWithGradAndHess],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_faffineErrorWithGradAndHess].rewind();
                
                /*
                Because each of the following kernels is synchronized by barriers on the GPU they can only be executed on a single compute device.
                Therefore, using an async queue may increase the speed by allowing multiple kernels to execute in parallel (if local memory permits)
                */
                localWorkSize = halfReductionSize % optimalMultiples[KERNEL_faffineSumInLocalMemoryCombined] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_faffineSumInLocalMemoryCombined] - (halfReductionSize % optimalMultiples[KERNEL_faffineSumInLocalMemoryCombined])), maximumElementsForLocalFPTcombinedSum);
                globalWorkSize = localWorkSize;
                uniformBSplineTransformProgramKernels[KERNEL_faffineSumInLocalMemoryCombined]
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
                        .putNullArg(localWorkSize*4 /*size in bytes of local mem allocation*/)
                        .putArg((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_faffineSumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_faffineSumInLocalMemoryCombined].rewind();
                
              //TODO: the implicit if's for float/double can be reduced by separating the cases into two blocks using a single if statement
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
                
                buffer = queue.putMapBuffer(gradient3, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                gradient[3] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(gradient3, buffer);
                
                buffer = queue.putMapBuffer(gradient4, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                gradient[4] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(gradient4, buffer);
                
                buffer = queue.putMapBuffer(gradient5, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                gradient[5] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(gradient5, buffer);
                
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
                
                buffer = queue.putMapBuffer(hessian03, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[0][3] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian03, buffer);
                
                buffer = queue.putMapBuffer(hessian04, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[0][4] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian04, buffer);
                
                buffer = queue.putMapBuffer(hessian05, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[0][5] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian05, buffer);
                
                buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[1][1] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian11, buffer);
                
                buffer = queue.putMapBuffer(hessian12, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[1][2] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian12, buffer);
                
                buffer = queue.putMapBuffer(hessian13, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[1][3] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian13, buffer);
                
                buffer = queue.putMapBuffer(hessian14, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[1][4] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian14, buffer);
                
                buffer = queue.putMapBuffer(hessian15, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[1][5] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian15, buffer);
                
                buffer = queue.putMapBuffer(hessian22, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[2][2] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian22, buffer);
                
                buffer = queue.putMapBuffer(hessian23, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[2][3] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian23, buffer);
                
                buffer = queue.putMapBuffer(hessian24, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[2][4] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian24, buffer);
                
                buffer = queue.putMapBuffer(hessian25, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[2][5] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian25, buffer);
                
                buffer = queue.putMapBuffer(hessian33, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[3][3] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian33, buffer);
                
                buffer = queue.putMapBuffer(hessian34, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[3][4] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian34, buffer);
                
                buffer = queue.putMapBuffer(hessian35, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[3][5] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian35, buffer);
                
                buffer = queue.putMapBuffer(hessian44, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[4][4] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian44, buffer);
                
                buffer = queue.putMapBuffer(hessian45, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[4][5] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian45, buffer);
                
                buffer = queue.putMapBuffer(hessian55, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[5][5] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian55, buffer);
            }
            else
            {
                
                int nrOfBlocks = ((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))%blockSizesParallel;
                nrOfBlocks = nrOfBlocks == 0 ? (int)Math.min(((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))/blockSizesParallel,maximumSumReductionBlockNr) : (int)Math.min((((int)(sourcePyramid[pyramidIndex].width * sourcePyramid[pyramidIndex].height))+(blockSizesParallel-nrOfBlocks))/blockSizesParallel,maximumSumReductionBlockNr);
                
                int globalWorkSize = nrOfBlocks * blockSizesParallel;
                uniformBSplineTransformProgramKernels[KERNEL_faffineErrorWithGradAndHessBrent]
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
                
                uniformBSplineTransformProgramKernels[KERNEL_faffineErrorWithGradAndHessBrent]
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/)
                        .putNullArg(blockSizesParallel*4 /*size in bytes of local mem allocation*/);
                
                uniformBSplineTransformProgramKernels[KERNEL_faffineErrorWithGradAndHessBrent]
                        .putArg((int)sourcePyramid[pyramidIndex].width)
                        .putArg((int)sourcePyramid[pyramidIndex].height)
                        .putArg((int)targetPyramid[pyramidIndex].width)
                        .putArg((int)targetPyramid[pyramidIndex].height);

                uniformBSplineTransformProgramKernels[KERNEL_faffineErrorWithGradAndHessBrent]
                		.putArg((float)currentoffsetx)
                        .putArg((float)curentoffsety)
                        .putArg((float)currenta11)
                        .putArg((float)currenta12)
                        .putArg((float)currenta21)
                        .putArg((float)currenta22);
               
                uniformBSplineTransformProgramKernels[KERNEL_faffineErrorWithGradAndHessBrent]
                        .putArg((int)targetPyramid[pyramidIndex].width*2)
                        .putArg((int)targetPyramid[pyramidIndex].height*2);
                
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_faffineErrorWithGradAndHessBrent],0,globalWorkSize,blockSizesParallel);
                uniformBSplineTransformProgramKernels[KERNEL_faffineErrorWithGradAndHessBrent].rewind();
                
                // use in-memory summations (also async) each with its own buffer to do the final reduction)
                halfReductionSize = (int) ((nrOfBlocks+(nrOfBlocks%2))/2);
                int localWorkSize = halfReductionSize % optimalMultiples[KERNEL_faffineSumInLocalMemoryCombined] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_faffineSumInLocalMemoryCombined] - (halfReductionSize % optimalMultiples[KERNEL_faffineSumInLocalMemoryCombined])), maximumElementsForLocalFPTcombinedSum);
                globalWorkSize = localWorkSize;
                uniformBSplineTransformProgramKernels[KERNEL_faffineSumInLocalMemoryCombined]
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
                        .putNullArg(halfReductionSize*4 /*size in bytes of local mem allocation*/)
                        .putArg((int)(nrOfBlocks));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_faffineSumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_faffineSumInLocalMemoryCombined].rewind();
                
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
                
                buffer = queue.putMapBuffer(gradient3, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                gradient[3] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(gradient3, buffer);
                
                buffer = queue.putMapBuffer(gradient4, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                gradient[4] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(gradient4, buffer);
                
                buffer = queue.putMapBuffer(gradient5, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                gradient[5] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(gradient5, buffer);
                        
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
                
                buffer = queue.putMapBuffer(hessian03, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[0][3] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian03, buffer);
                
                buffer = queue.putMapBuffer(hessian04, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[0][4] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian04, buffer);
                
                buffer = queue.putMapBuffer(hessian05, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[0][5] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian05, buffer);
                        
                buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[1][1] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian11, buffer);
                        
                buffer = queue.putMapBuffer(hessian12, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[1][2] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian12, buffer);
                
                buffer = queue.putMapBuffer(hessian13, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[1][3] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian13, buffer);
                
                buffer = queue.putMapBuffer(hessian14, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[1][4] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian14, buffer);
                
                buffer = queue.putMapBuffer(hessian15, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[1][5] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian15, buffer);
                        
                buffer = queue.putMapBuffer(hessian22, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[2][2] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian22, buffer);
                
                buffer = queue.putMapBuffer(hessian23, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[2][3] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian23, buffer);
                
                buffer = queue.putMapBuffer(hessian24, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[2][4] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian24, buffer);
                
                buffer = queue.putMapBuffer(hessian25, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[2][5] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian25, buffer);
                
                buffer = queue.putMapBuffer(hessian33, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[3][3] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian33, buffer);
                
                buffer = queue.putMapBuffer(hessian34, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[3][4] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian34, buffer);
                
                buffer = queue.putMapBuffer(hessian35, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[3][5] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian35, buffer);
                
                buffer = queue.putMapBuffer(hessian44, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[4][4] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian44, buffer);
                
                buffer = queue.putMapBuffer(hessian45, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[4][5] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian45, buffer);
                
                buffer = queue.putMapBuffer(hessian55, CLMemory.Map.READ, 0, 4, true);
                buffer.rewind();
                hessian[5][5] = (double)buffer.asFloatBuffer().get();
                queue.putUnmapMemory(hessian55, buffer);
            }
            //symmetrize hessian
            for (int i = 1; (i < 6); i++) {
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
            Arrays.fill(gradient, 0.0);
            for (int i = 0; (i < 2); i++) {
                Arrays.fill(hessian[i], 0.0);
            }
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
                            .putArg((int)targetDoubleSlice.width * 2)
                            .putArg((int)targetDoubleSlice.height * 2)
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

                    int nrOfBlocks = ((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))%doubleblockSizesParallel;
                    nrOfBlocks = nrOfBlocks == 0 ? (int)Math.min(((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))/doubleblockSizesParallel,maximumSumReductionBlockNr) : (int)Math.min((((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))+(doubleblockSizesParallel-nrOfBlocks))/doubleblockSizesParallel,maximumSumReductionBlockNr);

                    int globalWorkSize = nrOfBlocks * doubleblockSizesParallel;
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
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/);

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
                    queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dtranslationErrorWithGradAndHessBrent],0,globalWorkSize,doubleblockSizesParallel);
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
                final int width = (int)sharedContext.img.dimension(0);
                final int height = (int)sharedContext.img.dimension(1);
                final int doubletargetwidth = width * 2;
                final int doubletargetheight = height * 2;
                int nIndex = 0;
                int larea = 0;
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
                            
                            PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubletargetwidth, width, xInterpolationIndices);
                            PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubletargetheight, height, width, yInterpolationIndices);

                            rescoordx = PlainJavaCPUAligner.getFractional(coordx);
                            rescoordy = PlainJavaCPUAligner.getFractional(coordy);
                            
                            s = PlainJavaCPUAligner.interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, targetCoefficientDoubleSlice);

                            // calculate the values for returning
                            double diff = sourceImageDoubleSlice[nIndex] - s;// repurposed for diff
                            mse += diff * diff;
                            /*
                            TODO/FIXME/KNOWN ISSUE:
                            The following summation is MUCH worse than the parallel sum reduction done on the GPU, because (relatively speaking)
                            small numbers are added to an ever growing larger number reducing the precision in the outcome. Currently
                            I ignore this like the original implementation, but this is one of many reasons why the GPU version and the CPU version
                            will never yield the same results!
                            */
                            gradient[0] += diff * sourcexGradientDoubleSlice[nIndex];
                            gradient[1] += diff * sourceyGradientDoubleSlice[nIndex];
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
        
        private double doubleGetRigidBodyMeanSquares(double currentoffsetx, double currentoffsety, double currentangle)
        {
            double area = 0.0;
            double mse = 0.0;
            // First reset the global values which will not be reset in the loop
            Arrays.fill(gradient, 0.0);
            for (int i = 0; (i < 3); i++) {
                Arrays.fill(hessian[i], 0.0);
            }
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
                            .putArg((int)targetDoubleSlice.width * 2)
                            .putArg((int)targetDoubleSlice.height * 2)
                            .putArg(currentoffsetx)
                            .putArg(currentoffsety)
                            .putArg(Math.cos(currentangle))
                            .putArg(-Math.sin(currentangle));

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

                    int nrOfBlocks = ((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))%doubleblockSizesParallel;
                    nrOfBlocks = nrOfBlocks == 0 ? (int)Math.min(((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))/doubleblockSizesParallel,maximumSumReductionBlockNr) : (int)Math.min((((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))+(doubleblockSizesParallel-nrOfBlocks))/doubleblockSizesParallel,maximumSumReductionBlockNr);

                    int globalWorkSize = nrOfBlocks * doubleblockSizesParallel;
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
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/);

                    uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHessBrent]
                            .putArg((int)sourceDoubleSlice.width)
                            .putArg((int)sourceDoubleSlice.height)
                            .putArg((int)targetDoubleSlice.width)
                            .putArg((int)targetDoubleSlice.height);

                    uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHessBrent]
                            .putArg(currentoffsetx)
                            .putArg(currentoffsety)
                            .putArg(Math.cos(currentangle))
                            .putArg(-Math.sin(currentangle));

                    uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHessBrent]
                            .putArg((int)targetDoubleSlice.width*2)
                            .putArg((int)targetDoubleSlice.height*2);                
                    queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_drigidBodyErrorWithGradAndHessBrent],0,globalWorkSize,doubleblockSizesParallel);
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
                final int width = (int)sharedContext.img.dimension(0);
                final int height = (int)sharedContext.img.dimension(1);
                final int doubletargetwidth = width * 2;
                final int doubletargetheight = height * 2;
                int nIndex = 0;
                int larea = 0;
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
                            
                            PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubletargetwidth, width, xInterpolationIndices);
                            PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubletargetheight, height, width, yInterpolationIndices);

                            rescoordx = PlainJavaCPUAligner.getFractional(coordx);
                            rescoordy = PlainJavaCPUAligner.getFractional(coordy);
                            
                            s = PlainJavaCPUAligner.interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, targetCoefficientDoubleSlice);

                            // calculate the values for returning
                            double diff = sourceImageDoubleSlice[nIndex] - s;
                            mse += diff * diff;
                            double theta = sourceyGradientDoubleSlice[nIndex] * (double)n - sourcexGradientDoubleSlice[nIndex] * (double)i;
                            /*
                            TODO/FIXME/KNOWN ISSUE:
                            The following summation is MUCH worse than the parallel sum reduction done on the GPU, because (relatively speaking)
                            small numbers are added to an ever growing larger number reducing the precision in the outcome. Currently
                            I ignore this like the original implementation, but this is one of many reasons why the GPU version and the CPU version
                            will never yield the same results!
                            */
                            gradient[0] += diff * theta;
                            gradient[1] += diff * sourcexGradientDoubleSlice[nIndex];
                            gradient[2] += diff * sourceyGradientDoubleSlice[nIndex];
                            hessian[0][0] += theta * theta;
                            hessian[0][1] += theta * sourcexGradientDoubleSlice[nIndex];
                            hessian[0][2] += theta * sourceyGradientDoubleSlice[nIndex];
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
        
        private double doubleGetScaledRotationMeanSquares(double currentoffsetx, double currentoffsety, double currentangle, double currentscale)
        {
            double area = 0.0;
            double mse = 0.0;
            // First reset the global values which will not be reset in the loop
            Arrays.fill(gradient, 0.0);
            for (int i = 0; (i < 4); i++) {
                Arrays.fill(hessian[i], 0.0);
            }
            if(!usesFloat)
            {
                // now we have the diffs let's calculate all the derived elements
                int halfReductionSize = (int) (((sourceDoubleSlice.width * sourceDoubleSlice.height)+((sourceDoubleSlice.width * sourceDoubleSlice.height)%2))/2);
                ByteBuffer buffer;
                // calculate the MSE row-wise
                if(halfReductionSize <= doublemaximumElementsForLocalFPTcombinedSum)
                {                
                    uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationErrorWithGradAndHess]
                            .putArg(sourceDoubleSlice.Image)
                            .putArg(targetDoubleSlice.Coefficient)
                            .putArg(sourceDoubleSlice.xGradient)
                            .putArg(sourceDoubleSlice.yGradient)
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
                            .putArg(doubleEntryImageBuffer)
                            .putArg(maskBuffer)
                            .putArg((int)sourceDoubleSlice.width)
                            .putArg((int)sourceDoubleSlice.height)
                            .putArg((int)targetDoubleSlice.width)
                            .putArg((int)targetDoubleSlice.height)
                            .putArg((int)targetDoubleSlice.width * 2)
                            .putArg((int)targetDoubleSlice.height * 2)
                            .putArg(currentoffsetx)
                            .putArg(currentoffsety)
                            .putArg(Math.cos(currentangle))
                            .putArg(-Math.sin(currentangle))
                            .putArg(currentscale);

                    int localWorkSize = (int)Math.min(uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationErrorWithGradAndHess].getWorkGroupSize(device),blocksizeMultiplier*optimalMultiples[KERNEL_dscaledRotationErrorWithGradAndHess]);  // Local work size dimensions
                    int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dscaledRotationErrorWithGradAndHess], (int)(sourceDoubleSlice.width*sourceDoubleSlice.height));
                    queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationErrorWithGradAndHess],0,globalWorkSize,localWorkSize);
                    uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationErrorWithGradAndHess].rewind();

                    /*
                    Because each of the following kernels is synchronized by barriers on the GPU they can only be executed on a single compute device.
                    Therefore using an async queue may increase the speed by allowing multiple kernels to execute in parallel (if local memory permits)
                    */
                    localWorkSize = halfReductionSize % optimalMultiples[KERNEL_dscaledRotationSumInLocalMemoryCombined] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_dscaledRotationSumInLocalMemoryCombined] - (halfReductionSize % optimalMultiples[KERNEL_dscaledRotationSumInLocalMemoryCombined])), doublemaximumElementsForLocalFPTcombinedSum);
                    globalWorkSize = localWorkSize;
                    uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationSumInLocalMemoryCombined]
                            .putArg(maskBuffer)
                            .putArg(doubleEntryImageBuffer)
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
                            .putNullArg(localWorkSize*8 /*size in bytes of local mem allocation*/)
                            .putArg((int)(sourceDoubleSlice.width * sourceDoubleSlice.height));
                    queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationSumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                    uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationSumInLocalMemoryCombined].rewind();

                    buffer = queue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    area = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(maskBuffer, buffer);
                    buffer = queue.putMapBuffer(entryImageBuffer, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    mse = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(entryImageBuffer, buffer);
                    buffer = queue.putMapBuffer(gradient0, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[0] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient0, buffer);
                    buffer = queue.putMapBuffer(gradient1, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[1] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient1, buffer);
                    buffer = queue.putMapBuffer(gradient2, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[2] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient2, buffer);
                    buffer = queue.putMapBuffer(gradient3, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[3] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient3, buffer);
                    buffer = queue.putMapBuffer(hessian00, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][0] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian00, buffer);
                    buffer = queue.putMapBuffer(hessian01, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][1] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian01, buffer);
                    buffer = queue.putMapBuffer(hessian02, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][2] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian02, buffer);
                    buffer = queue.putMapBuffer(hessian03, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][3] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian03, buffer);
                    buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[1][1] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian11, buffer);
                    buffer = queue.putMapBuffer(hessian12, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[1][2] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian12, buffer);
                    buffer = queue.putMapBuffer(hessian13, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[1][3] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian13, buffer);
                    buffer = queue.putMapBuffer(hessian22, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[2][2] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian22, buffer);
                    buffer = queue.putMapBuffer(hessian23, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[2][3] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian23, buffer);
                    buffer = queue.putMapBuffer(hessian33, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[3][3] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian33, buffer);
                }
                else
                {

                    int nrOfBlocks = ((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))%doubleblockSizesParallel;
                    nrOfBlocks = nrOfBlocks == 0 ? (int)Math.min(((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))/doubleblockSizesParallel,maximumSumReductionBlockNr) : (int)Math.min((((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))+(doubleblockSizesParallel-nrOfBlocks))/doubleblockSizesParallel,maximumSumReductionBlockNr);

                    int globalWorkSize = nrOfBlocks * doubleblockSizesParallel;
                    uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationErrorWithGradAndHessBrent]
                            .putArg(sourceDoubleSlice.Image)
                            .putArg(targetDoubleSlice.Coefficient)
                            .putArg(sourceDoubleSlice.xGradient)
                            .putArg(sourceDoubleSlice.yGradient)
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
                            .putArg(doubleEntryImageBuffer)
                            .putArg(maskBuffer);


                    uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationErrorWithGradAndHessBrent]
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/);

                    uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationErrorWithGradAndHessBrent]
                            .putArg((int)sourceDoubleSlice.width)
                            .putArg((int)sourceDoubleSlice.height)
                            .putArg((int)targetDoubleSlice.width)
                            .putArg((int)targetDoubleSlice.height);

                    uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationErrorWithGradAndHessBrent]
                            .putArg(currentoffsetx)
                            .putArg(currentoffsety)
                            .putArg(Math.cos(currentangle))
                            .putArg(-Math.sin(currentangle))
                            .putArg(currentscale);

                    uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationErrorWithGradAndHessBrent]
                            .putArg((int)targetDoubleSlice.width*2)
                            .putArg((int)targetDoubleSlice.height*2);                
                    queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationErrorWithGradAndHessBrent],0,globalWorkSize,doubleblockSizesParallel);
                    uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationErrorWithGradAndHessBrent].rewind();

                    // use in-memory summations (also async) each with its own buffer to do the final reduction)
                    halfReductionSize = (int) ((nrOfBlocks+(nrOfBlocks%2))/2);
                    int localWorkSize = halfReductionSize % optimalMultiples[KERNEL_dscaledRotationSumInLocalMemoryCombined] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_dscaledRotationSumInLocalMemoryCombined] - (halfReductionSize % optimalMultiples[KERNEL_dscaledRotationSumInLocalMemoryCombined])), doublemaximumElementsForLocalFPTcombinedSum);
                    globalWorkSize = localWorkSize;
                    uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationSumInLocalMemoryCombined]
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
                            .putArg(doubleEntryImageBuffer)
                            .putArg(maskBuffer)
                            .putNullArg(halfReductionSize*8 /*size in bytes of local mem allocation*/)
                            .putArg((int)(nrOfBlocks));
                    queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationSumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                    uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationSumInLocalMemoryCombined].rewind();

                    buffer = queue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    area = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(maskBuffer, buffer);
                    
                    buffer = queue.putMapBuffer(entryImageBuffer, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    mse = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(entryImageBuffer, buffer);
                            
                    buffer = queue.putMapBuffer(gradient0, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[0] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient0, buffer);
                            
                    buffer = queue.putMapBuffer(gradient1, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[1] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient1, buffer);
                            
                    buffer = queue.putMapBuffer(gradient2, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[2] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient2, buffer);
                    
                    buffer = queue.putMapBuffer(gradient3, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[3] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient3, buffer);
                            
                    buffer = queue.putMapBuffer(hessian00, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][0] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian00, buffer);
                            
                    buffer = queue.putMapBuffer(hessian01, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][1] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian01, buffer);
                            
                    buffer = queue.putMapBuffer(hessian02, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][2] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian02, buffer);
                    
                    buffer = queue.putMapBuffer(hessian03, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][3] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian03, buffer);
                            
                    buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[1][1] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian11, buffer);
                            
                    buffer = queue.putMapBuffer(hessian12, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[1][2] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian12, buffer);
                    
                    buffer = queue.putMapBuffer(hessian13, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[1][3] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian13, buffer);
                            
                    buffer = queue.putMapBuffer(hessian22, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[2][2] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian22, buffer);
                    
                    buffer = queue.putMapBuffer(hessian23, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[2][3] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian23, buffer);
                    
                    buffer = queue.putMapBuffer(hessian33, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[3][3] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian33, buffer);
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
                double s;
                double xvecx = Math.cos(currentangle) * currentscale;
                double xvecy = -Math.sin(currentangle) * currentscale;
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
                            
                            PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubletargetwidth, width, xInterpolationIndices);
                            PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubletargetheight, height, width, yInterpolationIndices);

                            rescoordx = PlainJavaCPUAligner.getFractional(coordx);
                            rescoordy = PlainJavaCPUAligner.getFractional(coordy);
                            
                            s = PlainJavaCPUAligner.interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, targetCoefficientDoubleSlice);

                            // calculate the values for returning
                            double diff = sourceImageDoubleSlice[nIndex] - s;
                            mse += diff * diff;
                            double theta = sourceyGradientDoubleSlice[nIndex] * (double)n - sourcexGradientDoubleSlice[nIndex] * (double)i;
                            double j_scale = (((double)n) * sourcexGradientDoubleSlice[nIndex] + ((double)i) * sourceyGradientDoubleSlice[nIndex]); // scale contribution to j
                            /*
                            TODO/FIXME/KNOWN ISSUE:
                            The following summation is MUCH worse than the parallel sum reduction done on the GPU, because (relatively speaking)
                            small numbers are added to an ever growing larger number reducing the precision in the outcome. Currently
                            I ignore this like the original implementation, but this is one of many reasons why the GPU version and the CPU version
                            will never yield the same results!
                            */
                            gradient[0] += diff * j_scale;
                            gradient[1] += diff * theta;
                            gradient[2] += diff * sourcexGradientDoubleSlice[nIndex];
                            gradient[3] += diff * sourceyGradientDoubleSlice[nIndex];
                            hessian[0][0] += j_scale * j_scale;
                            hessian[0][1] += j_scale * theta;
                            hessian[0][2] += j_scale * sourcexGradientDoubleSlice[nIndex];
                            hessian[0][3] += j_scale * sourceyGradientDoubleSlice[nIndex];
                            hessian[1][1] += theta * theta;
                            hessian[1][2] += theta * sourcexGradientDoubleSlice[nIndex];
                            hessian[1][3] += theta * sourceyGradientDoubleSlice[nIndex];
                            hessian[2][2] += sourcexGradientDoubleSlice[nIndex] * sourcexGradientDoubleSlice[nIndex];
                            hessian[2][3] += sourcexGradientDoubleSlice[nIndex] * sourceyGradientDoubleSlice[nIndex];
                            hessian[3][3] += sourceyGradientDoubleSlice[nIndex] * sourceyGradientDoubleSlice[nIndex];
                        }
                        // walk along the X-vector direction
                        coordx += xvecx;
                        coordy += xvecy;
                    }
                }
                area = (double)larea;
            }
            // symmetrize hessian
            for (int i = 1; (i < 4); i++) {
                for (int j = 0; (j < i); j++) {
                        hessian[i][j] = hessian[j][i];
                }
            }
            return mse/area;
        }
        
        private double doubleGetAffineMeanSquares(double currentoffsetx, double currentoffsety,
                double currenta11, double currenta12,
                double currenta21, double currenta22)
        {
            double area = 0.0;
            double mse = 0.0;
            // First reset the global values which will not be reset in the loop
            Arrays.fill(gradient, 0.0);
            for (int i = 0; i < 6; i++) {
                Arrays.fill(hessian[i], 0.0);
            }
            if(!usesFloat)
            {
                // now we have the diffs let's calculate all the derived elements
                int halfReductionSize = (int) (((sourceDoubleSlice.width * sourceDoubleSlice.height)+((sourceDoubleSlice.width * sourceDoubleSlice.height)%2))/2);
                ByteBuffer buffer;
                // calculate the MSE row-wise
                if(halfReductionSize <= doublemaximumElementsForLocalFPTcombinedSum)
                {                
                    uniformBSplineTransformProgramKernels[KERNEL_daffineErrorWithGradAndHess]
                            .putArg(sourceDoubleSlice.Image)
                            .putArg(targetDoubleSlice.Coefficient)
                            .putArg(sourceDoubleSlice.xGradient)
                            .putArg(sourceDoubleSlice.yGradient)
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
                            .putArg(doubleEntryImageBuffer)
                            .putArg((int)sourceDoubleSlice.width)
                            .putArg((int)sourceDoubleSlice.height)
                            .putArg((int)targetDoubleSlice.width)
                            .putArg((int)targetDoubleSlice.height)
                            .putArg((int)targetDoubleSlice.width * 2)
                            .putArg((int)targetDoubleSlice.height * 2)
                            .putArg(currentoffsetx)
                            .putArg(currentoffsety)
                            .putArg(currenta11)
                            .putArg(currenta12)
                            .putArg(currenta21)
                            .putArg(currenta22);

                    int localWorkSize = (int)Math.min(uniformBSplineTransformProgramKernels[KERNEL_daffineErrorWithGradAndHess].getWorkGroupSize(device),blocksizeMultiplier*optimalMultiples[KERNEL_daffineErrorWithGradAndHess]);  // Local work size dimensions
                    int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_daffineErrorWithGradAndHess], (int)(sourceDoubleSlice.width*sourceDoubleSlice.height));
                    queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_daffineErrorWithGradAndHess],0,globalWorkSize,localWorkSize);
                    uniformBSplineTransformProgramKernels[KERNEL_daffineErrorWithGradAndHess].rewind();

                    /*
                    Because each of the following kernels is synchronized by barriers on the GPU they can only be executed on a single compute device.
                    Therefore using an async queue may increase the speed by allowing multiple kernels to execute in parallel (if local memory permits)
                    */
                    localWorkSize = halfReductionSize % optimalMultiples[KERNEL_daffineSumInLocalMemoryCombined] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_daffineSumInLocalMemoryCombined] - (halfReductionSize % optimalMultiples[KERNEL_daffineSumInLocalMemoryCombined])), doublemaximumElementsForLocalFPTcombinedSum);
                    globalWorkSize = localWorkSize;
                    uniformBSplineTransformProgramKernels[KERNEL_daffineSumInLocalMemoryCombined]
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
                            .putNullArg(localWorkSize*8 /*size in bytes of local mem allocation*/)
                            .putArg((int)(sourceDoubleSlice.width * sourceDoubleSlice.height));
                    queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_daffineSumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                    uniformBSplineTransformProgramKernels[KERNEL_daffineSumInLocalMemoryCombined].rewind();

                  //TODO: the implicit if's for float/double can be reduced by separating the cases into two blocks using a single if statement
                    buffer = queue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    area = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(maskBuffer, buffer);
                    
                    buffer = queue.putMapBuffer(entryImageBuffer, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    mse = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(entryImageBuffer, buffer);
                    
                    buffer = queue.putMapBuffer(gradient0, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[0] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient0, buffer);
                    
                    buffer = queue.putMapBuffer(gradient1, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[1] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient1, buffer);
                    
                    buffer = queue.putMapBuffer(gradient2, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[2] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient2, buffer);
                    
                    buffer = queue.putMapBuffer(gradient3, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[3] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient3, buffer);
                    
                    buffer = queue.putMapBuffer(gradient4, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[4] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient4, buffer);
                    
                    buffer = queue.putMapBuffer(gradient5, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[5] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient5, buffer);
                    
                    buffer = queue.putMapBuffer(hessian00, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][0] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian00, buffer);
                    
                    buffer = queue.putMapBuffer(hessian01, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][1] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian01, buffer);
                    
                    buffer = queue.putMapBuffer(hessian02, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][2] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian02, buffer);
                    
                    buffer = queue.putMapBuffer(hessian03, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][3] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian03, buffer);
                    
                    buffer = queue.putMapBuffer(hessian04, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][4] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian04, buffer);
                    
                    buffer = queue.putMapBuffer(hessian05, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][5] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian05, buffer);
                    
                    buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[1][1] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian11, buffer);
                    
                    buffer = queue.putMapBuffer(hessian12, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[1][2] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian12, buffer);
                    
                    buffer = queue.putMapBuffer(hessian13, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[1][3] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian13, buffer);
                    
                    buffer = queue.putMapBuffer(hessian14, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[1][4] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian14, buffer);
                    
                    buffer = queue.putMapBuffer(hessian15, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[1][5] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian15, buffer);
                    
                    buffer = queue.putMapBuffer(hessian22, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[2][2] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian22, buffer);
                    
                    buffer = queue.putMapBuffer(hessian23, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[2][3] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian23, buffer);
                    
                    buffer = queue.putMapBuffer(hessian24, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[2][4] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian24, buffer);
                    
                    buffer = queue.putMapBuffer(hessian25, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[2][5] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian25, buffer);
                    
                    buffer = queue.putMapBuffer(hessian33, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[3][3] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian33, buffer);
                    
                    buffer = queue.putMapBuffer(hessian34, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[3][4] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian34, buffer);
                    
                    buffer = queue.putMapBuffer(hessian35, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[3][5] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian35, buffer);
                    
                    buffer = queue.putMapBuffer(hessian44, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[4][4] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian44, buffer);
                    
                    buffer = queue.putMapBuffer(hessian45, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[4][5] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian45, buffer);
                    
                    buffer = queue.putMapBuffer(hessian55, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[5][5] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian55, buffer);
                }
                else
                {

                    int nrOfBlocks = ((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))%doubleblockSizesParallel;
                    nrOfBlocks = nrOfBlocks == 0 ? (int)Math.min(((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))/doubleblockSizesParallel,maximumSumReductionBlockNr) : (int)Math.min((((int)(sourceDoubleSlice.width * sourceDoubleSlice.height))+(doubleblockSizesParallel-nrOfBlocks))/doubleblockSizesParallel,maximumSumReductionBlockNr);

                    int globalWorkSize = nrOfBlocks * doubleblockSizesParallel;
                    uniformBSplineTransformProgramKernels[KERNEL_daffineErrorWithGradAndHessBrent]
                            .putArg(sourceDoubleSlice.Image)
                            .putArg(targetDoubleSlice.Coefficient)
                            .putArg(sourceDoubleSlice.xGradient)
                            .putArg(sourceDoubleSlice.yGradient)
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


                    uniformBSplineTransformProgramKernels[KERNEL_daffineErrorWithGradAndHessBrent]
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/)
                            .putNullArg(doubleblockSizesParallel*8 /*size in bytes of local mem allocation*/);

                    uniformBSplineTransformProgramKernels[KERNEL_daffineErrorWithGradAndHessBrent]
                            .putArg((int)sourceDoubleSlice.width)
                            .putArg((int)sourceDoubleSlice.height)
                            .putArg((int)targetDoubleSlice.width)
                            .putArg((int)targetDoubleSlice.height);

                    uniformBSplineTransformProgramKernels[KERNEL_daffineErrorWithGradAndHessBrent]
                    		.putArg(currentoffsetx)
                            .putArg(currentoffsety)
                            .putArg(currenta11)
                            .putArg(currenta12)
                            .putArg(currenta21)
                            .putArg(currenta22);

                    uniformBSplineTransformProgramKernels[KERNEL_daffineErrorWithGradAndHessBrent]
                            .putArg((int)targetDoubleSlice.width*2)
                            .putArg((int)targetDoubleSlice.height*2);                
                    queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_daffineErrorWithGradAndHessBrent],0,globalWorkSize,doubleblockSizesParallel);
                    uniformBSplineTransformProgramKernels[KERNEL_daffineErrorWithGradAndHessBrent].rewind();

                    // use in-memory summations (also async) each with its own buffer to do the final reduction)
                    halfReductionSize = (int) ((nrOfBlocks+(nrOfBlocks%2))/2);
                    int localWorkSize = halfReductionSize % optimalMultiples[KERNEL_daffineSumInLocalMemoryCombined] == 0 ? halfReductionSize : (int)Math.min(halfReductionSize + (optimalMultiples[KERNEL_daffineSumInLocalMemoryCombined] - (halfReductionSize % optimalMultiples[KERNEL_daffineSumInLocalMemoryCombined])), doublemaximumElementsForLocalFPTcombinedSum);
                    globalWorkSize = localWorkSize;
                    uniformBSplineTransformProgramKernels[KERNEL_daffineSumInLocalMemoryCombined]
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
                            .putNullArg(halfReductionSize*8 /*size in bytes of local mem allocation*/)
                            .putArg((int)(nrOfBlocks));
                    queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_daffineSumInLocalMemoryCombined],0,globalWorkSize,localWorkSize);
                    uniformBSplineTransformProgramKernels[KERNEL_daffineSumInLocalMemoryCombined].rewind();

                    buffer = queue.putMapBuffer(maskBuffer, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    area = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(maskBuffer, buffer);
                    
                    buffer = queue.putMapBuffer(entryImageBuffer, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    mse = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(entryImageBuffer, buffer);
                            
                    buffer = queue.putMapBuffer(gradient0, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[0] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient0, buffer);
                            
                    buffer = queue.putMapBuffer(gradient1, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[1] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient1, buffer);
                            
                    buffer = queue.putMapBuffer(gradient2, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[2] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient2, buffer);
                    
                    buffer = queue.putMapBuffer(gradient3, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[3] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient3, buffer);
                    
                    buffer = queue.putMapBuffer(gradient4, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[4] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient4, buffer);
                    
                    buffer = queue.putMapBuffer(gradient5, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    gradient[5] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(gradient5, buffer);
                            
                    buffer = queue.putMapBuffer(hessian00, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][0] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian00, buffer);
                            
                    buffer = queue.putMapBuffer(hessian01, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][1] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian01, buffer);
                            
                    buffer = queue.putMapBuffer(hessian02, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][2] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian02, buffer);
                    
                    buffer = queue.putMapBuffer(hessian03, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][3] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian03, buffer);
                    
                    buffer = queue.putMapBuffer(hessian04, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][4] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian04, buffer);
                    
                    buffer = queue.putMapBuffer(hessian05, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[0][5] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian05, buffer);
                            
                    buffer = queue.putMapBuffer(hessian11, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[1][1] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian11, buffer);
                            
                    buffer = queue.putMapBuffer(hessian12, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[1][2] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian12, buffer);
                    
                    buffer = queue.putMapBuffer(hessian13, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[1][3] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian13, buffer);
                    
                    buffer = queue.putMapBuffer(hessian14, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[1][4] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian14, buffer);
                    
                    buffer = queue.putMapBuffer(hessian15, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[1][5] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian15, buffer);
                            
                    buffer = queue.putMapBuffer(hessian22, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[2][2] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian22, buffer);
                    
                    buffer = queue.putMapBuffer(hessian23, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[2][3] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian23, buffer);
                    
                    buffer = queue.putMapBuffer(hessian24, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[2][4] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian24, buffer);
                    
                    buffer = queue.putMapBuffer(hessian25, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[2][5] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian25, buffer);
                    
                    buffer = queue.putMapBuffer(hessian33, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[3][3] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian33, buffer);
                    
                    buffer = queue.putMapBuffer(hessian34, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[3][4] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian34, buffer);
                    
                    buffer = queue.putMapBuffer(hessian35, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[3][5] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian35, buffer);
                    
                    buffer = queue.putMapBuffer(hessian44, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[4][4] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian44, buffer);
                    
                    buffer = queue.putMapBuffer(hessian45, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[4][5] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian45, buffer);
                    
                    buffer = queue.putMapBuffer(hessian55, CLMemory.Map.READ, 0, 8, true);
                    buffer.rewind();
                    hessian[5][5] = (double)buffer.asDoubleBuffer().get();
                    queue.putUnmapMemory(hessian55, buffer);
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
                double s;
                double coordx;
                double rescoordx;
                double coordy;
                double rescoordy;
                int mskx;
                int msky;
                
                /*
                 * Affine coordinate vectors:
                 *   xvec = (a11, a21) — the column-step direction (stepping n by +1)
                 *   yvec = (a12, a22) — the row-step direction (stepping i by +1)
                 *
                 * For output pixel (n, i):
                 *   coordx = tx + n*a11 + i*a12
                 *   coordy = ty + n*a21 + i*a22
                 *
                 * We use incremental stepping (adding xvec per column, resetting
                 * to base + i*yvec per row) to avoid per-pixel multiplications,
                 * identical to how rigid body uses cos/sin vectors.
                 */
                double xvecx = currenta11;
                double xvecy = currenta21;
                double yvecx = currenta12;
                double yvecy = currenta22;
                
                /*
                 * Jacobian column variables (recomputed per pixel):
                 *   dx0 = n * ∂f/∂x   (∂f/∂a11)
                 *   dx1 = i * ∂f/∂x   (∂f/∂a12)
                 *   dy0 = n * ∂f/∂y   (∂f/∂a21)
                 *   dy1 = i * ∂f/∂y   (∂f/∂a22)
                 *   dx  = ∂f/∂x       (∂f/∂tx)
                 *   dy  = ∂f/∂y       (∂f/∂ty)
                 */
                double dx0, dx1, dy0, dy1, dx, dy;
                
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
                            
                            PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubletargetwidth, width, xInterpolationIndices);
                            PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubletargetheight, height, width, yInterpolationIndices);

                            rescoordx = PlainJavaCPUAligner.getFractional(coordx);
                            rescoordy = PlainJavaCPUAligner.getFractional(coordy);
                            
                            s = PlainJavaCPUAligner.interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, targetCoefficientDoubleSlice);

                            // calculate the values for returning
                            double diff = sourceImageDoubleSlice[nIndex] - s;
                            mse += diff * diff;
                            
                            /*
                             * Jacobian columns for the affine parameterization:
                             *   ∂f/∂a11 = n * ∂f/∂x       → dx0
                             *   ∂f/∂a12 = i * ∂f/∂x       → dx1
                             *   ∂f/∂a21 = n * ∂f/∂y       → dy0
                             *   ∂f/∂a22 = i * ∂f/∂y       → dy1
                             *   ∂f/∂tx  = ∂f/∂x           → dx
                             *   ∂f/∂ty  = ∂f/∂y           → dy
                             *
                             * These use the SOURCE gradients (the "inverse" approach).
                             */
                            dx = sourcexGradientDoubleSlice[nIndex];
                            dy = sourceyGradientDoubleSlice[nIndex];
                            dx0 = ((double)n) * dx;   // n * ∂f/∂x
                            dx1 = ((double)i) * dx;   // i * ∂f/∂x
                            dy0 = ((double)n) * dy;   // n * ∂f/∂y
                            dy1 = ((double)i) * dy;   // i * ∂f/∂y

                            /*
                             * Gradient accumulation: g_k += r_i * J_ik
                             * Parameter order: (a11, a12, a21, a22, tx, ty)
                             */
                            gradient[0] += diff * dx0;   // Σ r * n * ∂f/∂x
                            gradient[1] += diff * dx1;   // Σ r * i * ∂f/∂x
                            gradient[2] += diff * dy0;   // Σ r * n * ∂f/∂y
                            gradient[3] += diff * dy1;   // Σ r * i * ∂f/∂y
                            gradient[4] += diff * dx;    // Σ r * ∂f/∂x
                            gradient[5] += diff * dy;    // Σ r * ∂f/∂y

                            /*
                             * Hessian accumulation (upper triangle only):
                             * H_kl += J_ik * J_il
                             *
                             * This is the Gauss-Newton approximation H ≈ J^T J,
                             * ignoring the second-order terms r_i * ∇²f_i.
                             * Only 21 unique entries (upper triangle of 6×6 symmetric matrix).
                             */
                            // Row 0: a11 × {a11, a12, a21, a22, tx, ty}
                            hessian[0][0] += dx0 * dx0;
                            hessian[0][1] += dx0 * dx1;
                            hessian[0][2] += dx0 * dy0;
                            hessian[0][3] += dx0 * dy1;
                            hessian[0][4] += dx0 * dx;
                            hessian[0][5] += dx0 * dy;
                            // Row 1: a12 × {a12, a21, a22, tx, ty}
                            hessian[1][1] += dx1 * dx1;
                            hessian[1][2] += dx1 * dy0;
                            hessian[1][3] += dx1 * dy1;
                            hessian[1][4] += dx1 * dx;
                            hessian[1][5] += dx1 * dy;
                            // Row 2: a21 × {a21, a22, tx, ty}
                            hessian[2][2] += dy0 * dy0;
                            hessian[2][3] += dy0 * dy1;
                            hessian[2][4] += dy0 * dx;
                            hessian[2][5] += dy0 * dy;
                            // Row 3: a22 × {a22, tx, ty}
                            hessian[3][3] += dy1 * dy1;
                            hessian[3][4] += dy1 * dx;
                            hessian[3][5] += dy1 * dy;
                            // Row 4: tx × {tx, ty}
                            hessian[4][4] += dx * dx;
                            hessian[4][5] += dx * dy;
                            // Row 5: ty × {ty}
                            hessian[5][5] += dy * dy;
                        }
                        // walk along the X-vector direction
                        coordx += xvecx;
                        coordy += xvecy;
                    }
                }
                area = (double)larea;
            }
            // symmetrize hessian
            for (int i = 1; (i < 6); i++) {
                for (int j = 0; (j < i); j++) {
                        hessian[i][j] = hessian[j][i];
                }
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
                        .putArg((int)targetDoubleSlice.width * 2)
                        .putArg((int)targetDoubleSlice.height * 2)
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
                            
                            PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubletargetwidth, width, xInterpolationIndices);
                            PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubletargetheight, height, width, yInterpolationIndices);

                            rescoordx = PlainJavaCPUAligner.getFractional(coordx);
                            rescoordy = PlainJavaCPUAligner.getFractional(coordy);
                            
                            s = PlainJavaCPUAligner.interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, targetCoefficientDoubleSlice);


                            // calculate the MSE
                            double diff = sourceImageDoubleSlice[nIndex] - s;
                            mse += diff * diff;
                        }
                        // walk along the X-vector direction
                        coordx += 1.0;
                    }
                }
                area = (double)larea;
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
                        .putArg((int)targetDoubleSlice.width * 2)
                        .putArg((int)targetDoubleSlice.height * 2)
                        .putArg(currentoffsetx)
                        .putArg(currentoffsety)
                        .putArg(Math.cos(currentangle))
                        .putArg(-Math.sin(currentangle));          
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
                            
                            PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubletargetwidth, width, xInterpolationIndices);
                            PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubletargetheight, height, width, yInterpolationIndices);

                            rescoordx = PlainJavaCPUAligner.getFractional(coordx);
                            rescoordy = PlainJavaCPUAligner.getFractional(coordy);
                            
                            s = PlainJavaCPUAligner.interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, targetCoefficientDoubleSlice);

                            // calculate the MSE
                            double diff = sourceImageDoubleSlice[nIndex] - s;
                            mse += diff * diff;
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
        
        private double doubleGetScaledRotationMeanSquaresWithoutHessian(double currentoffsetx, double currentoffsety, double currentangle, double currentscale)
        {
            double area = 0.0;
            double mse = 0.0;
            if(!usesFloat)
            {
                uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationError]
                        .putArg(sourceDoubleSlice.Image)
                        .putArg(targetDoubleSlice.Coefficient)
                        .putArg(doubleEntryImageBuffer)
                        .putArg(maskBuffer)
                        .putArg((int)sourceDoubleSlice.width)
                        .putArg((int)sourceDoubleSlice.height)
                        .putArg((int)targetDoubleSlice.width)
                        .putArg((int)targetDoubleSlice.height)
                        .putArg((int)targetDoubleSlice.width * 2)
                        .putArg((int)targetDoubleSlice.height * 2)
                        .putArg(currentoffsetx)
                        .putArg(currentoffsety)
                        .putArg(Math.cos(currentangle))
                        .putArg(-Math.sin(currentangle))
                        .putArg(currentscale);          
                int localWorkSize = (int)Math.min(uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationError].getWorkGroupSize(device),optimalMultiples[KERNEL_dscaledRotationError]*blocksizeMultiplier);  // Local work size dimensions
                int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_dscaledRotationError], (int)(sourceDoubleSlice.width*sourceDoubleSlice.height));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationError],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_dscaledRotationError].rewind();
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
                double s;
                double xvecx = Math.cos(currentangle) * currentscale;
                double xvecy = -Math.sin(currentangle) * currentscale;
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
                            
                            PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubletargetwidth, width, xInterpolationIndices);
                            PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubletargetheight, height, width, yInterpolationIndices);

                            rescoordx = PlainJavaCPUAligner.getFractional(coordx);
                            rescoordy = PlainJavaCPUAligner.getFractional(coordy);
                            
                            s = PlainJavaCPUAligner.interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, targetCoefficientDoubleSlice);

                            // calculate the MSE
                            double diff = sourceImageDoubleSlice[nIndex] - s;
                            mse += diff * diff;
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
        
        private double doubleGetAffineMeanSquaresWithoutHessian(double currentoffsetx, double currentoffsety,
                double currenta11, double currenta12,
                double currenta21, double currenta22)
        {
            double area = 0.0;
            double mse = 0.0;
            if(!usesFloat)
            {
                uniformBSplineTransformProgramKernels[KERNEL_daffineError]
                        .putArg(sourceDoubleSlice.Image)
                        .putArg(targetDoubleSlice.Coefficient)
                        .putArg(doubleEntryImageBuffer)
                        .putArg(maskBuffer)
                        .putArg((int)sourceDoubleSlice.width)
                        .putArg((int)sourceDoubleSlice.height)
                        .putArg((int)targetDoubleSlice.width)
                        .putArg((int)targetDoubleSlice.height)
                        .putArg((int)targetDoubleSlice.width * 2)
                        .putArg((int)targetDoubleSlice.height * 2)
                        .putArg(currentoffsetx)
                        .putArg(currentoffsety)
                        .putArg(currenta11)
                        .putArg(currenta12)
                        .putArg(currenta21)
                        .putArg(currenta22);          
                int localWorkSize = (int)Math.min(uniformBSplineTransformProgramKernels[KERNEL_daffineError].getWorkGroupSize(device),optimalMultiples[KERNEL_daffineError]*blocksizeMultiplier);  // Local work size dimensions
                int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_daffineError], (int)(sourceDoubleSlice.width*sourceDoubleSlice.height));
                queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_daffineError],0,globalWorkSize,localWorkSize);
                uniformBSplineTransformProgramKernels[KERNEL_daffineError].rewind();
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
                double s;
                double coordx;
                double rescoordx;
                double coordy;
                double rescoordy;
                int mskx;
                int msky;
                
                // Affine coordinate step vectors (same as getAffineMeanSquares)
                double xvecx = currenta11;
                double xvecy = currenta21;
                double yvecx = currenta12;
                double yvecy = currenta22;
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
                            
                            PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubletargetwidth, width, xInterpolationIndices);
                            PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubletargetheight, height, width, yInterpolationIndices);

                            rescoordx = PlainJavaCPUAligner.getFractional(coordx);
                            rescoordy = PlainJavaCPUAligner.getFractional(coordy);
                            
                            s = PlainJavaCPUAligner.interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, targetCoefficientDoubleSlice);

                            // calculate the MSE
                            double diff = sourceImageDoubleSlice[nIndex] - s;
                            mse += diff * diff;
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
                    .putArg((int)targetPyramid[pyramidIndex].width * 2)
                    .putArg((int)targetPyramid[pyramidIndex].height * 2)
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
                    .putArg((int)targetPyramid[pyramidIndex].height * 2)
                    .putArg((float)currentoffsetx)
                    .putArg((float)curentoffsety)
                    .putArg((float)Math.cos(currentangle))
                    .putArg((float)-Math.sin(currentangle));
            
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
        
        private double getScaledRotationMeanSquaresWithoutHessian(int pyramidIndex, double currentoffsetx, double currentoffsety, double currentangle, double currentscale)
        {
            uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationError]
                    .putArg(sourcePyramid[pyramidIndex].Image)
                    .putArg(targetPyramid[pyramidIndex].Coefficient)
                    .putArg(entryImageBuffer)
                    .putArg(maskBuffer)
                    .putArg((int)sourcePyramid[pyramidIndex].width)
                    .putArg((int)sourcePyramid[pyramidIndex].height)
                    .putArg((int)targetPyramid[pyramidIndex].width)
                    .putArg((int)targetPyramid[pyramidIndex].height)
                    .putArg((int)targetPyramid[pyramidIndex].width * 2)
                    .putArg((int)targetPyramid[pyramidIndex].height * 2)
                    .putArg((float)currentoffsetx)
                    .putArg((float)currentoffsety)
                    .putArg((float)Math.cos(currentangle))
                    .putArg((float)-Math.sin(currentangle))
                    .putArg((float)currentscale);
            
            int localWorkSize = (int)Math.min(uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationError].getWorkGroupSize(device),optimalMultiples[KERNEL_fscaledRotationError]*blocksizeMultiplier);  // Local work size dimensions
            int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_fscaledRotationError], (int)(sourcePyramid[pyramidIndex].width*sourcePyramid[pyramidIndex].height));
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationError],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_fscaledRotationError].rewind();
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
        
        private double getAffineMeanSquaresWithoutHessian(int pyramidIndex, double currentoffsetx, double currentoffsety,
                double currenta11, double currenta12,
                double currenta21, double currenta22)
        {
            uniformBSplineTransformProgramKernels[KERNEL_faffineError]
                    .putArg(sourcePyramid[pyramidIndex].Image)
                    .putArg(targetPyramid[pyramidIndex].Coefficient)
                    .putArg(entryImageBuffer)
                    .putArg(maskBuffer)
                    .putArg((int)sourcePyramid[pyramidIndex].width)
                    .putArg((int)sourcePyramid[pyramidIndex].height)
                    .putArg((int)targetPyramid[pyramidIndex].width)
                    .putArg((int)targetPyramid[pyramidIndex].height)
                    .putArg((int)targetPyramid[pyramidIndex].width * 2)
                    .putArg((int)targetPyramid[pyramidIndex].height * 2)
                    .putArg((float)currentoffsetx)
                    .putArg((float)currentoffsety)
                    .putArg((float)currenta11)
                    .putArg((float)currenta12)
                    .putArg((float)currenta21)
                    .putArg((float)currenta22);
            
            int localWorkSize = (int)Math.min(uniformBSplineTransformProgramKernels[KERNEL_faffineError].getWorkGroupSize(device),optimalMultiples[KERNEL_faffineError]*blocksizeMultiplier);  // Local work size dimensions
            int globalWorkSize = StaticUtility.roundUp(localWorkSize, optimalMultiples[KERNEL_faffineError], (int)(sourcePyramid[pyramidIndex].width*sourcePyramid[pyramidIndex].height));
            queue.put1DRangeKernel(uniformBSplineTransformProgramKernels[KERNEL_faffineError],0,globalWorkSize,localWorkSize);
            uniformBSplineTransformProgramKernels[KERNEL_faffineError].rewind();
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
