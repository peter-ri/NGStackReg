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

import ch.unibas.biozentrum.imagejplugins.abstracts.CPUAligner;
import ch.unibas.biozentrum.imagejplugins.abstracts.ImageConverter;
import ch.unibas.biozentrum.imagejplugins.util.AffineTransformation;
import ch.unibas.biozentrum.imagejplugins.util.RigidBodyTransformation;
import ch.unibas.biozentrum.imagejplugins.util.ScaledRotationTransformation;
import ch.unibas.biozentrum.imagejplugins.util.TranslationTransformation;
import ch.unibas.biozentrum.imagejplugins.util.StaticUtility;

import java.util.Arrays;
import java.util.concurrent.BrokenBarrierException;

/**
 * @author Peter D. Ringel
 * @version 1.0.0
 *
 */
class FloatMTNGSourcePyramidSlice {
    long width;
    long height;
    float[] Image;
    float[] xGradient;
    float[] yGradient;
}

class FloatMTNGTargetPyramidSlice {
    long width;
    long height;
    float[] Coefficient;
}

public class HybridPrecisionCPUAligner extends CPUAligner {

    public static final float pole = -0.26794919243112270647255365849413f;
    public static final float[] polesDeg7 = { -0.5352804307964381655424037816816460718339231523426924148812f,
                                              -0.122554615192326690515272264359357343605486549427295558490763f,
                                              -0.0091486948096082769285930216516478534156925639545994482648003f };
    public static final float h0D3 = 0.66666666666666666666666666666666666666666666666666666666666666667f;
    public static final float h1D3 = 0.16666666666666666666666666666666666666666666666666666666666666667f;
    public static final float h0D7 = 0.4793650793650793650793650793650793650793650793650793650793650793651f;
    public static final float h1D7 = 0.23630952380952380952380952380952380952380952380952380952380952380952f;
    public static final float h2D7 = 0.023809523809523809523809523809523809523809523809523809523809523810f;
    public static final float h3D7 = 0.00019841269841269841269841269841269841269841269841269841269841269841f;
    public static final float lambda7 = 5040.0f;
    public static final float rh0 = 0.375f;
    public static final float rh1 = 0.25f;
    public static final float rh2 = 0.0625f;

    boolean fwd = true;
    private double offsetx = 0.0;
    private double offsety = 0.0;
    private double angle = 0.0;
    private double scale = 1.0;
    private double a11 = 1.0;
    private double a12 = 0.0;
    private double a21 = 0.0;
    private double a22 = 1.0;
    private final double[][] hessian;
    private final float[][] fhessian;
    private final double[][] pseudoHessian;
    private final double[] gradient;
    private final float[] fgradient;
    private final int[] xInterpolationIndices;
    private final int[] yInterpolationIndices;
    private int iterationPower;

    private FloatMTNGSourcePyramidSlice[] sourcePyramid;
    private FloatMTNGTargetPyramidSlice[] targetPyramid;

    private float[] entryImageBuffers;
    private float[] fullSizedHelperBuffer;

    private double[] doublefullSizedHelperBuffer;
    private double[] doublefullSizedHelperBuffer2;

    private double[] doubleSourceImage;
    private double[] doubleSourcexGradient;
    private double[] doubleSourceyGradient;
    private double[] doubleTargetCoefficients;

    private final SharedContextAlignmentTarget scat = new SharedContextAlignmentTarget();

    HybridPrecisionCPUAligner(final AbstractSharedContext sharedContext, final ImageConverter converter,
            final int pyramidDepth) {
        super(sharedContext, converter, pyramidDepth);
        this.xInterpolationIndices = new int[] { 0, 0, 0, 0 };
        this.yInterpolationIndices = new int[] { 0, 0, 0, 0 };
        
        switch(sharedContext.transformationType) {
        case TRANSLATION:
            this.gradient = new double[]{0.0, 0.0};
            this.fgradient = new float[] { 0.0f, 0.0f };
            this.hessian = new double[][]{{0.0,0.0}, {0.0,0.0}};
            this.fhessian = new float[][] { { 0.0f, 0.0f }, { 0.0f, 0.0f } };
            this.pseudoHessian = new double[][]{{0.0,0.0}, {0.0,0.0}};
            break;
        case RIGIDBODY:
            this.gradient = new double[]{0.0, 0.0, 0.0};
            this.fgradient = new float[] { 0.0f, 0.0f, 0.0f };
            this.hessian = new double[][]{{0.0,0.0,0.0}, {0.0,0.0,0.0}, {0.0,0.0,0.0}};
            this.fhessian = new float[][] { { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f } };
            this.pseudoHessian = new double[][]{{0.0,0.0,0.0}, {0.0,0.0,0.0}, {0.0,0.0,0.0}};
            break;
        case SCALEDROTATION:
            this.gradient = new double[]{0.0, 0.0, 0.0, 0.0};
            this.fgradient = new float[] { 0.0f, 0.0f, 0.0f, 0.0f };
            this.hessian = new double[][]{{0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0}};
            this.fhessian = new float[][] { { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } };
            this.pseudoHessian = new double[][]{{0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0}};
            break;
        case AFFINE:
            this.gradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            this.fgradient = new float[] { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
            this.hessian = new double[][]{{0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}};
            this.fhessian = new float[][] { { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f } };
            this.pseudoHessian = new double[][]{{0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}};
            break;
        default:
            this.gradient = new double[]{0.0, 0.0, 0.0};
            this.fgradient = new float[] { 0.0f, 0.0f, 0.0f };
            this.hessian = new double[][]{{0.0,0.0,0.0}, {0.0,0.0,0.0}, {0.0,0.0,0.0}};
            this.fhessian = new float[][] { { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f } };
            this.pseudoHessian = new double[][]{{0.0,0.0,0.0}, {0.0,0.0,0.0}, {0.0,0.0,0.0}};
            break;
        }
        
        allocateMemory();
    }

    private void allocateMemory() {
        long width = sharedContext.img.dimension(0);
        long height = sharedContext.img.dimension(1);
        if (width * height > Integer.MAX_VALUE) {
            throw new RuntimeException("Cannot allocate more than " + Integer.MAX_VALUE);
        }
        sourcePyramid = new FloatMTNGSourcePyramidSlice[pyramidDepth];
        targetPyramid = new FloatMTNGTargetPyramidSlice[pyramidDepth];
        entryImageBuffers = new float[(int) (width * height)];
        fullSizedHelperBuffer = new float[(int) (width * height)];
        doublefullSizedHelperBuffer = new double[(int) (width * height)];
        doubleSourceImage = new double[(int) (width * height)];
        doubleSourcexGradient = new double[(int) (width * height)];
        doubleSourceyGradient = new double[(int) (width * height)];
        doubleTargetCoefficients = new double[(int) (width * height)];
        doublefullSizedHelperBuffer2 = new double[(int) (width * height)];
        for (int j = 0; j < pyramidDepth; j++) {
            sourcePyramid[j] = new FloatMTNGSourcePyramidSlice();
            targetPyramid[j] = new FloatMTNGTargetPyramidSlice();
            sourcePyramid[j].width = width;
            sourcePyramid[j].height = height;
            sourcePyramid[j].Image = new float[(int) (width * height)];
            sourcePyramid[j].xGradient = new float[(int) (width * height)];
            sourcePyramid[j].yGradient = new float[(int) (width * height)];
            targetPyramid[j].width = width;
            targetPyramid[j].height = height;
            targetPyramid[j].Coefficient = new float[(int) (width * height)];
            width /= 2;
            height /= 2;
        }
    }
    
    private void resizeAllocatedBuffers() {
        long width = sharedContext.resizedTargetImage.dimension(0);
        long height = sharedContext.resizedTargetImage.dimension(1);
        if (width * height > Integer.MAX_VALUE) {
            throw new RuntimeException("Cannot allocate more than " + Integer.MAX_VALUE);
        }
        for (int j = 0; j < pyramidDepth; j++) {
            sourcePyramid[j].Image = null;
            sourcePyramid[j].xGradient = null;
            sourcePyramid[j].yGradient = null;
            targetPyramid[j].Coefficient = null;
        }
        sourcePyramid = null;
        targetPyramid = null;
        entryImageBuffers = null;
        fullSizedHelperBuffer = null;
        doubleSourceImage = null;
        doubleSourcexGradient = null;
        doubleSourceyGradient = null;
        doubleTargetCoefficients = null;
        doublefullSizedHelperBuffer2 = new double[(int) (width * height)];
    }

    @Override
    public void run() {
        while (!sharedContext.getNextAlignmentTarget(scat)) {
            // Construct the target pyramid
            constructTargetImagePyramid();
            // Construct the source image and derivative pyramids
            constructSourceImagePyramid();
            // Now that we have the pyramid the optimization can commence
            doRegistration();
            switch (sharedContext.transformationType) {
            case TRANSLATION:
                ((TranslationTransformation)scat.transformation).offsetx = offsetx;
                ((TranslationTransformation)scat.transformation).offsety = offsety;
                // Don't forget to reset the transformation values for the next image
                offsetx = 0.0;
                offsety = 0.0;
                break;
            case RIGIDBODY:
                ((RigidBodyTransformation) scat.transformation).angle = angle;
                ((RigidBodyTransformation) scat.transformation).offsetx = offsetx;
                ((RigidBodyTransformation) scat.transformation).offsety = offsety;
                // Don't forget to reset the transformation values for the next image
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
        }
        try {
            // Done calculating the transformations now wait until all threads are done
            sharedContext.workerSynchronizationBarrier.await(); // This also combines all the transformations and resets
                                                                // the position (see the action implementation)
        } catch (InterruptedException ex) {
            throw new RuntimeException("Thread interrupted.");
        } catch (BrokenBarrierException ex) {
            throw new RuntimeException("Worker synchronization barrier is broken.");
        }
        
        if(sharedContext instanceof SharedContextZT) 
        {
            while (!sharedContext.getNextAlignmentTarget(scat)) {
                // Construct the target pyramid
                constructTargetImagePyramid();
                // Construct the source image and derivative pyramids
                constructSourceImagePyramid();
                // Now that we have the pyramid the optimization can commence
                doRegistration();
                switch (sharedContext.transformationType) {
                case TRANSLATION:
                    ((TranslationTransformation)scat.transformation).offsetx = offsetx;
                    ((TranslationTransformation)scat.transformation).offsety = offsety;
                    // reset the transformation values for the next image
                    offsetx = 0.0;
                    offsety = 0.0;
                    angle = 0.0;
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
            }
            try {
                // Done calculating the transformations now wait until all threads are done
                sharedContext.workerSynchronizationBarrier.await(); // This also combines all the transformations and resets
                                                                    // the position (see the action implementation)
            } catch (InterruptedException ex) {
                throw new RuntimeException("Thread interrupted.");
            } catch (BrokenBarrierException ex) {
                throw new RuntimeException("Worker synchronization barrier is broken.");
            }
        }
        
        if(sharedContext.getResizeAfterRegistration())
        {
            resizeAllocatedBuffers();
            // The following code requires the current position to have been reset to a 0vector
            while (!sharedContext.getTransformationForCurrentPosition(scat)) {
                int width = (int) sharedContext.img.dimension(0);
                int height = (int) sharedContext.img.dimension(1);
                converter.convertTo(scat.targetArray, doublefullSizedHelperBuffer);
                // pre-multiply the image for cubic spline interpolation
                PlainJavaCPUAligner.premultiplyCubicBSpline(doublefullSizedHelperBuffer, width * height);
                // Conversion to B-spline coefficients along X axis
                PlainJavaCPUAligner.cubicBSplinePrefilter2DXhp(doublefullSizedHelperBuffer, width, height);
                // pre-multiply again
                PlainJavaCPUAligner.premultiplyCubicBSpline(doublefullSizedHelperBuffer, width * height);
                // Now along the Y-axis
                PlainJavaCPUAligner.cubicBSplinePrefilter2DYhp(doublefullSizedHelperBuffer, width, height);
                switch (sharedContext.transformationType) {
                    case TRANSLATION:
                        resizeTransformTranslateImageWithBsplineInterpolation(width, height, (int) sharedContext.resizedTargetImage.dimension(0), (int) sharedContext.resizedTargetImage.dimension(1), ((TranslationTransformation) scat.transformation).offsetx,
                                ((TranslationTransformation) scat.transformation).offsety);
                        break;
                    case RIGIDBODY:
                        resizeTransformImageWithBsplineInterpolation(width, height, (int) sharedContext.resizedTargetImage.dimension(0), (int) sharedContext.resizedTargetImage.dimension(1), ((RigidBodyTransformation) scat.transformation).offsetx,
                                ((RigidBodyTransformation) scat.transformation).offsety,
                                ((RigidBodyTransformation) scat.transformation).angle);
                        break;
                    case SCALEDROTATION:
                        resizeScaledRotationImageWithBsplineInterpolation(width, height, (int)sharedContext.resizedTargetImage.dimension(0), (int)sharedContext.resizedTargetImage.dimension(1), ((ScaledRotationTransformation)scat.transformation).offsetx, ((ScaledRotationTransformation)scat.transformation).offsety, ((ScaledRotationTransformation)scat.transformation).angle, ((ScaledRotationTransformation)scat.transformation).scale);
                        break;
                    case AFFINE:
                        resizeAffineImageWithBsplineInterpolation(width, height, (int)sharedContext.resizedTargetImage.dimension(0), (int)sharedContext.resizedTargetImage.dimension(1), ((AffineTransformation)scat.transformation).offsetx, ((AffineTransformation)scat.transformation).offsety, ((AffineTransformation)scat.transformation).a11, ((AffineTransformation)scat.transformation).a12, ((AffineTransformation)scat.transformation).a21, ((AffineTransformation)scat.transformation).a22);
                        break;
                }
                converter.deConvertTo(doublefullSizedHelperBuffer2, scat.sourceArray);
            }
        }
        else
        {
            // The following code requires the current position to have been reset to a 0vector
            while (!sharedContext.getTransformationForCurrentPosition(scat)) {
                int width = (int) sharedContext.img.dimension(0);
                int height = (int) sharedContext.img.dimension(1);
                converter.convertTo(scat.targetArray, doublefullSizedHelperBuffer);
                // pre-multiply the image for cubic spline interpolation
                PlainJavaCPUAligner.premultiplyCubicBSpline(doublefullSizedHelperBuffer, width * height);
                // Conversion to B-spline coefficients along X axis
                PlainJavaCPUAligner.cubicBSplinePrefilter2DXhp(doublefullSizedHelperBuffer, width, height);
                // pre-multiply again
                PlainJavaCPUAligner.premultiplyCubicBSpline(doublefullSizedHelperBuffer, width * height);
                // Now along the Y-axis
                PlainJavaCPUAligner.cubicBSplinePrefilter2DYhp(doublefullSizedHelperBuffer, width, height);
                switch (sharedContext.transformationType) {
                case TRANSLATION:
                    transformTranslateImageWithBsplineInterpolation(width, height, ((TranslationTransformation) scat.transformation).offsetx,
                            ((TranslationTransformation) scat.transformation).offsety);
                    break;
                case RIGIDBODY:
                    transformImageWithBsplineInterpolation(width, height, ((RigidBodyTransformation) scat.transformation).offsetx,
                            ((RigidBodyTransformation) scat.transformation).offsety,
                            ((RigidBodyTransformation) scat.transformation).angle);
                    break;
                case SCALEDROTATION:
                    transformScaledRotationWithBsplineInterpolation(width,height, ((ScaledRotationTransformation)scat.transformation).offsetx, ((ScaledRotationTransformation)scat.transformation).offsety, ((ScaledRotationTransformation)scat.transformation).angle, ((ScaledRotationTransformation)scat.transformation).scale);
                    break;
                case AFFINE:
                    transformAffineWithBsplineInterpolation(width,height, ((AffineTransformation)scat.transformation).offsetx, ((AffineTransformation)scat.transformation).offsety, ((AffineTransformation)scat.transformation).a11, ((AffineTransformation)scat.transformation).a12, ((AffineTransformation)scat.transformation).a21, ((AffineTransformation)scat.transformation).a22);
                    break;
                }
                converter.deConvertTo(doublefullSizedHelperBuffer2, scat.targetArray);
            }
        }
        
    }
    
    private void transformTranslateImageWithBsplineInterpolation(final int width, final int height, double currentoffsetx,
            double currentoffsety) {
        /*
         * Requires the coefficients to be in entryImageBuffers and the output will be
         * in fullSizedHelperBuffer
         */
        int doubleWidth = width * 2;
        int doubleHeight = height * 2;
        int nIndex = 0;
        double coordx;
        double coordy;
        int mskx;
        int msky;
        for (int i = 0; i < height; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx;
            coordy = currentoffsety + ((double) i);
            for (int n = 0; n < width; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < width) && (msky >= 0) && (msky < height)) {
                    PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                    PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                    
                    doublefullSizedHelperBuffer2[nIndex] = PlainJavaCPUAligner.interpolateCubicBSpline(PlainJavaCPUAligner.getFractional(coordx), PlainJavaCPUAligner.getFractional(coordy), xInterpolationIndices, yInterpolationIndices, doublefullSizedHelperBuffer);
                } else {
                    doublefullSizedHelperBuffer2[nIndex] = 0.0;
                }
                // now walk along the X-vector direction
                coordx += 1.0;
            }
        }
    }
    
    private void resizeTransformTranslateImageWithBsplineInterpolation(final int width, final int height, final int targetwidth, final int targetheight, double currentoffsetx,
            double currentoffsety) {
        /*
         * Requires the coefficients to be in entryImageBuffers and the output will be
         * in fullSizedHelperBuffer
         */
        int doubleWidth = width * 2;
        int doubleHeight = height * 2;
        int nIndex = 0;
        double coordx;
        double coordy;
        int mskx;
        int msky;
        for (int i = 0; i < targetheight; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx;
            coordy = currentoffsety + ((double) i);
            for (int n = 0; n < targetwidth; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < width) && (msky >= 0) && (msky < height)) {
                	PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                    PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                    
                    doublefullSizedHelperBuffer2[nIndex] = PlainJavaCPUAligner.interpolateCubicBSpline(PlainJavaCPUAligner.getFractional(coordx), PlainJavaCPUAligner.getFractional(coordy), xInterpolationIndices, yInterpolationIndices, doublefullSizedHelperBuffer);
                } else {
                    doublefullSizedHelperBuffer2[nIndex] = 0.0;
                }
                // now walk along the X-vector direction
                coordx += 1.0;
            }
        }
    }

    private void transformImageWithBsplineInterpolation(final int width, final int height, double currentoffsetx,
            double currentoffsety, double currentangle) {
        /*
         * Requires the coefficients to be in entryImageBuffers and the output will be
         * in fullSizedHelperBuffer
         */
        int doubleWidth = width * 2;
        int doubleHeight = height * 2;
        int nIndex = 0;
        double xvecx = Math.cos(currentangle);
        double xvecy = -Math.sin(currentangle);
        double yvecx = -xvecy;
        double yvecy = xvecx;
        double coordx;
        double coordy;
        int mskx;
        int msky;
        for (int i = 0; i < height; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx + ((double) i) * yvecx;
            coordy = currentoffsety + ((double) i) * yvecy;
            for (int n = 0; n < width; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < width) && (msky >= 0) && (msky < height)) {
                	PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                    PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                    
                    doublefullSizedHelperBuffer2[nIndex] = PlainJavaCPUAligner.interpolateCubicBSpline(PlainJavaCPUAligner.getFractional(coordx), PlainJavaCPUAligner.getFractional(coordy), xInterpolationIndices, yInterpolationIndices, doublefullSizedHelperBuffer);
                } else {
                    doublefullSizedHelperBuffer2[nIndex] = 0.0;
                }
                // now walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
    }
    
    private void resizeTransformImageWithBsplineInterpolation(final int width, final int height, final int targetwidth, final int targetheight, double currentoffsetx,
            double currentoffsety, double currentangle) {
        /*
         * Requires the coefficients to be in entryImageBuffers and the output will be
         * in fullSizedHelperBuffer
         */
        int doubleWidth = width * 2;
        int doubleHeight = height * 2;
        int nIndex = 0;
        double xvecx = Math.cos(currentangle);
        double xvecy = -Math.sin(currentangle);
        double yvecx = -xvecy;
        double yvecy = xvecx;
        double coordx;
        double coordy;
        int mskx;
        int msky;
        for (int i = 0; i < targetheight; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx + ((double) i) * yvecx;
            coordy = currentoffsety + ((double) i) * yvecy;
            for (int n = 0; n < targetwidth; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < width) && (msky >= 0) && (msky < height)) {
                	PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                    PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                    
                    doublefullSizedHelperBuffer2[nIndex] = PlainJavaCPUAligner.interpolateCubicBSpline(PlainJavaCPUAligner.getFractional(coordx), PlainJavaCPUAligner.getFractional(coordy), xInterpolationIndices, yInterpolationIndices, doublefullSizedHelperBuffer);
                } else {
                    doublefullSizedHelperBuffer2[nIndex] = 0.0;
                }
                // now walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
    }
    
    private void transformScaledRotationWithBsplineInterpolation(final int width, final int height, double currentoffsetx,
            double currentoffsety, double currentangle, double currentscale) {
        /*
         * Requires the coefficients to be in entryImageBuffers and the output will be
         * in fullSizedHelperBuffer
         */
        int doubleWidth = width * 2;
        int doubleHeight = height * 2;
        int nIndex = 0;
        double xvecx = Math.cos(currentangle) * currentscale;
        double xvecy = -Math.sin(currentangle) * currentscale;
        double yvecx = -xvecy;
        double yvecy = xvecx;
        double coordx;
        double coordy;
        int mskx;
        int msky;
        for (int i = 0; i < height; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx + ((double) i) * yvecx;
            coordy = currentoffsety + ((double) i) * yvecy;
            for (int n = 0; n < width; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < width) && (msky >= 0) && (msky < height)) {
                	PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                    PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                    
                    doublefullSizedHelperBuffer2[nIndex] = PlainJavaCPUAligner.interpolateCubicBSpline(PlainJavaCPUAligner.getFractional(coordx), PlainJavaCPUAligner.getFractional(coordy), xInterpolationIndices, yInterpolationIndices, doublefullSizedHelperBuffer);
                } else {
                    doublefullSizedHelperBuffer2[nIndex] = 0.0;
                }
                // now walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
    }
    
    private void resizeScaledRotationImageWithBsplineInterpolation(final int width, final int height, final int targetwidth, final int targetheight, double currentoffsetx,
            double currentoffsety, double currentangle, double currentscale) {
        /*
         * Requires the coefficients to be in entryImageBuffers and the output will be
         * in fullSizedHelperBuffer
         */
        int doubleWidth = width * 2;
        int doubleHeight = height * 2;
        int nIndex = 0;
        double xvecx = Math.cos(currentangle) * currentscale;
        double xvecy = -Math.sin(currentangle) * currentscale;
        double yvecx = -xvecy;
        double yvecy = xvecx;
        double coordx;
        double coordy;
        int mskx;
        int msky;
        for (int i = 0; i < targetheight; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx + ((double) i) * yvecx;
            coordy = currentoffsety + ((double) i) * yvecy;
            for (int n = 0; n < targetwidth; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < width) && (msky >= 0) && (msky < height)) {
                	PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                    PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                    
                    doublefullSizedHelperBuffer2[nIndex] = PlainJavaCPUAligner.interpolateCubicBSpline(PlainJavaCPUAligner.getFractional(coordx), PlainJavaCPUAligner.getFractional(coordy), xInterpolationIndices, yInterpolationIndices, doublefullSizedHelperBuffer);
                } else {
                    doublefullSizedHelperBuffer2[nIndex] = 0.0;
                }
                // now walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
    }
    
    
    private void transformAffineWithBsplineInterpolation(final int width, final int height, double currentoffsetx,
            double currentoffsety, double currenta11, double currenta12, double currenta21, double currenta22) {
        /*
         * Requires the coefficients to be in entryImageBuffers and the output will be
         * in fullSizedHelperBuffer
         */
        int doubleWidth = width * 2;
        int doubleHeight = height * 2;
        int nIndex = 0;
        double xvecx = currenta11;
        double xvecy = currenta21;
        double yvecx = currenta12;
        double yvecy = currenta22;
        double coordx;
        double coordy;
        int mskx;
        int msky;
        for (int i = 0; i < height; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx + ((double) i) * yvecx;
            coordy = currentoffsety + ((double) i) * yvecy;
            for (int n = 0; n < width; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < width) && (msky >= 0) && (msky < height)) {
                	PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                    PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                    
                    doublefullSizedHelperBuffer2[nIndex] = PlainJavaCPUAligner.interpolateCubicBSpline(PlainJavaCPUAligner.getFractional(coordx), PlainJavaCPUAligner.getFractional(coordy), xInterpolationIndices, yInterpolationIndices, doublefullSizedHelperBuffer);
                } else {
                    doublefullSizedHelperBuffer2[nIndex] = 0.0;
                }
                // now walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
    }
    
    private void resizeAffineImageWithBsplineInterpolation(final int width, final int height, final int targetwidth, final int targetheight, double currentoffsetx,
            double currentoffsety, double currenta11, double currenta12, double currenta21, double currenta22) {
        /*
         * Requires the coefficients to be in entryImageBuffers and the output will be
         * in fullSizedHelperBuffer
         */
        int doubleWidth = width * 2;
        int doubleHeight = height * 2;
        int nIndex = 0;
        double xvecx = currenta11;
        double xvecy = currenta21;
        double yvecx = currenta12;
        double yvecy = currenta22;
        double coordx;
        double coordy;
        int mskx;
        int msky;
        for (int i = 0; i < targetheight; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx + ((double) i) * yvecx;
            coordy = currentoffsety + ((double) i) * yvecy;
            for (int n = 0; n < targetwidth; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < width) && (msky >= 0) && (msky < height)) {
                	PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                    PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                    
                    doublefullSizedHelperBuffer2[nIndex] = PlainJavaCPUAligner.interpolateCubicBSpline(PlainJavaCPUAligner.getFractional(coordx), PlainJavaCPUAligner.getFractional(coordy), xInterpolationIndices, yInterpolationIndices, doublefullSizedHelperBuffer);
                } else {
                    doublefullSizedHelperBuffer2[nIndex] = 0.0;
                }
                // now walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
    }
    
    private static void premultiplyCubicBSplineDeg7(final float target[], final int nrOfElements) {
        for (int i = 0; i < nrOfElements; i++) {
            target[i] *= lambda7;
        }
    }

    private static void cubicBSplinePrefilter2DXhpDeg7(final float target[], final int width, final int height) {
        for (int p = 0; p < 3; p++) {
            for (int i = 0; i < height; i++) {
                // causal initialization
                float z1 = polesDeg7[p];
                float zn = (float) Math.pow(z1, (double) width);
                float sum = (1.0f + polesDeg7[p]) * (target[i * width] + zn * target[i * width + (width - 1)]);
                zn *= zn;
                for (int n = 1; n < width - 1; n++) {
                    z1 *= polesDeg7[p];
                    zn /= polesDeg7[p];
                    sum += (z1 + zn) * target[i * width + n];
                }
                target[i * width] = (sum / (1.0f - (float) Math.pow(polesDeg7[p], (float) (2 * width))));
                // causal recursion
                for (int n = 1; n < width; n++) {
                    target[i * width + n] += polesDeg7[p] * target[i * width + n - 1];
                }
                // anticausal initialization
                target[i * width + (width - 1)] = (polesDeg7[p] * target[i * width + (width - 1)]
                        / (polesDeg7[p] - 1.0f));

                // anticausal recursion
                for (int n = width - 2; n >= 0; n--) {
                    target[i * width + n] = polesDeg7[p] * (target[i * width + n + 1] - target[i * width + n]);
                }
            }
        }
    }

    private static void cubicBSplinePrefilter2DYhpDeg7(final float target[], final int width, final int height) {
        for (int p = 0; p < 3; p++) {
            for (int i = 0; i < width; i++) {
                // causal initialization
                float z1 = polesDeg7[p];
                float zn = (float) Math.pow(z1, (double) height);
                float sum = (1.0f + polesDeg7[p]) * (target[i] + zn * target[(height - 1) * width + i]);
                zn *= zn;
                for (int n = 1; n < height - 1; n++) {
                    z1 *= polesDeg7[p];
                    zn /= polesDeg7[p];
                    sum += (z1 + zn) * target[n * width + i];
                }
                target[i] = (sum / (1.0f - (float) Math.pow(polesDeg7[p], (double) (2 * height))));
                // causal recursion
                for (int n = 1; n < height; n++) {
                    target[n * width + i] += polesDeg7[p] * target[(n - 1) * width + i];
                }
                // anticausal initialization
                target[(height - 1) * width + i] = (polesDeg7[p] * target[(height - 1) * width + i]
                        / (polesDeg7[p] - 1.0f));

                // anticausal recursion
                for (int n = height - 2; n >= 0; n--) {
                    target[n * width + i] = polesDeg7[p] * (target[(n + 1) * width + i] - target[n * width + i]);
                }
            }
        }
    }

    private static void basicToCardinal2DXhpDeg7(final float input[], final float output[], final int width,
            final int height) {
        // Mirroring conditions can safely be ignored because width >6 is guaranteed.
        // symmetricFirMirrorOffBounds1D
        for (int i = 0; i < height; i++) {
            int nIndex = i * width + 3;
            // Run for all non-border condition pixels to prevent checking the conditions
            // all the time.
            for (int n = 3; n < width - 3; n++, nIndex++) {
                output[nIndex] = h0D7 * input[nIndex] + h1D7 * (input[nIndex - 1] + input[nIndex + 1])
                        + h2D7 * (input[nIndex - 2] + input[nIndex + 2])
                        + h3D7 * (input[nIndex - 3] + input[nIndex + 3]);
            }
            // now the left boundary condition
            // n == 0
            nIndex = i * width;
            output[nIndex] = h0D7 * input[nIndex] + h1D7 * (input[nIndex] + input[nIndex + 1])
                    + h2D7 * (input[nIndex + 1] + input[nIndex + 2]) + h3D7 * (input[nIndex + 2] + input[nIndex + 3]);
            // n == 1
            nIndex++;
            output[nIndex] = h0D7 * input[nIndex] + h1D7 * (input[nIndex - 1] + input[nIndex + 1])
                    + h2D7 * (input[nIndex - 1] + input[nIndex + 2]) + h3D7 * (input[nIndex] + input[nIndex + 3]);
            // n == 2
            nIndex++;
            output[nIndex] = h0D7 * input[nIndex] + h1D7 * (input[nIndex - 1] + input[nIndex + 1])
                    + h2D7 * (input[nIndex - 2] + input[nIndex + 2]) + h3D7 * (input[nIndex - 2] + input[nIndex + 3]);
            // now the right boundary condition
            // n == width - 3
            nIndex = (i + 1) * width - 3;
            output[nIndex] = h0D7 * input[nIndex] + h1D7 * (input[nIndex - 1] + input[nIndex + 1])
                    + h2D7 * (input[nIndex - 2] + input[nIndex + 2]) + h3D7 * (input[nIndex - 3] + input[nIndex + 2]);
            // n == width - 2
            nIndex++;
            output[nIndex] = h0D7 * input[nIndex] + h1D7 * (input[nIndex - 1] + input[nIndex + 1])
                    + h2D7 * (input[nIndex - 2] + input[nIndex + 1]) + h3D7 * (input[nIndex - 3] + input[nIndex]);
            // n == width -1
            nIndex++;
            output[nIndex] = h0D7 * input[nIndex] + h1D7 * (input[nIndex - 1] + input[nIndex])
                    + h2D7 * (input[nIndex - 2] + input[nIndex - 1]) + h3D7 * (input[nIndex - 3] + input[nIndex - 2]);
        }
    }

    private static void basicToCardinal2DYhpDeg7(final float input[], final float output[], final int width,
            final int height) {
        // Mirroring conditions can safely be ignored because width >6 is guaranteed.
        // symmetricFirMirrorOffBounds1D
        for (int i = 0; i < width; i++) {
            int nIndex = 3 * width + i;
            // Run for all non-border condition pixels to prevent checking the conditions
            // all the time.
            for (int n = 3; n < height - 3; n++, nIndex += width) {
                output[nIndex] = h0D7 * input[nIndex] + h1D7 * (input[nIndex - width] + input[nIndex + width])
                        + h2D7 * (input[nIndex - 2 * width] + input[nIndex + 2 * width])
                        + h3D7 * (input[nIndex - 3 * width] + input[nIndex + 3 * width]);
            }
            // now the top boundary condition
            // n == 0
            nIndex = i;
            output[nIndex] = h0D7 * input[nIndex] + h1D7 * (input[nIndex] + input[nIndex + width])
                    + h2D7 * (input[nIndex + width] + input[nIndex + 2 * width])
                    + h3D7 * (input[nIndex + 2 * width] + input[nIndex + 3 * width]);
            // n == 1
            nIndex += width;
            output[nIndex] = h0D7 * input[nIndex] + h1D7 * (input[nIndex - width] + input[nIndex + width])
                    + h2D7 * (input[nIndex - width] + input[nIndex + 2 * width])
                    + h3D7 * (input[nIndex] + input[nIndex + 3 * width]);
            // n == 2
            nIndex += width;
            output[nIndex] = h0D7 * input[nIndex] + h1D7 * (input[nIndex - width] + input[nIndex + width])
                    + h2D7 * (input[nIndex - 2 * width] + input[nIndex + 2 * width])
                    + h3D7 * (input[nIndex - 2 * width] + input[nIndex + 3 * width]);
            // now the bottom boundary condition
            // n == height - 3
            nIndex = (height - 3) * width + i;
            output[nIndex] = h0D7 * input[nIndex] + h1D7 * (input[nIndex - width] + input[nIndex + width])
                    + h2D7 * (input[nIndex - 2 * width] + input[nIndex + 2 * width])
                    + h3D7 * (input[nIndex - 3 * width] + input[nIndex + 2 * width]);
            // n == height - 2
            nIndex += width;
            output[nIndex] = h0D7 * input[nIndex] + h1D7 * (input[nIndex - width] + input[nIndex + width])
                    + h2D7 * (input[nIndex - 2 * width] + input[nIndex + width])
                    + h3D7 * (input[nIndex - 3 * width] + input[nIndex]);
            // n == height -1
            nIndex += width;
            output[nIndex] = h0D7 * input[nIndex] + h1D7 * (input[nIndex - width] + input[nIndex])
                    + h2D7 * (input[nIndex - 2 * width] + input[nIndex - width])
                    + h3D7 * (input[nIndex - 3 * width] + input[nIndex - 2 * width]);
        }
    }

    private static void reduceDual1DX(final float input[], final float output[], final int width, final int height,
            final int halfwidth) {
        for (int i = 0; i < height; i++) {
            int nIndex = i * halfwidth + 1;
            int rIndex = i * width + 2;
            for (int n = 1; n < halfwidth - 1; n++, nIndex++, rIndex += 2) {
                output[nIndex] = rh0 * input[rIndex] + rh1 * (input[rIndex - 1] + input[rIndex + 1])
                        + rh2 * (input[rIndex - 2] + input[rIndex + 2]);
            }
            // now the mirror boundary conditions
            // n == 0
            nIndex = i * halfwidth;
            rIndex = i * width;
            output[nIndex] = rh0 * input[rIndex] + rh1 * (input[rIndex] + input[rIndex + 1])
                    + rh2 * (input[rIndex + 1] + input[rIndex + 2]);
            // n == halfwidth - 1
            nIndex = (i + 1) * halfwidth - 1;
            rIndex = (i + 1) * width - 2;
            if (width == (2 * halfwidth))// Yes this can be different if width % 2 != 0
            {
                output[nIndex] = rh0 * input[rIndex] + rh1 * (input[rIndex - 1] + input[rIndex + 1])
                        + rh2 * (input[rIndex - 2] + input[rIndex + 1]);
            } else {
                output[nIndex] = rh0 * input[rIndex - 1] + rh1 * (input[rIndex - 2] + input[rIndex])
                        + rh2 * (input[rIndex - 3] + input[rIndex + 1]);
            }
        }
    }

    private static void reduceDual1DY(final float input[], final float output[], final int halfwidth, final int height,
            final int halfheight) {
        for (int i = 0; i < halfwidth; i++) {
            int nIndex = i + halfwidth;
            int rIndex = i + 2 * halfwidth;
            for (int n = 1; n < halfheight - 1; n++, nIndex += halfwidth, rIndex += 2 * halfwidth) {
                output[nIndex] = rh0 * input[rIndex] + rh1 * (input[rIndex - halfwidth] + input[rIndex + halfwidth])
                        + rh2 * (input[rIndex - 2 * halfwidth] + input[rIndex + 2 * halfwidth]);
            }
            // now the mirror boundary conditions
            // n == 0
            nIndex = i;
            output[nIndex] = rh0 * input[nIndex] + rh1 * (input[nIndex] + input[nIndex + halfwidth])
                    + rh2 * (input[nIndex + halfwidth] + input[nIndex + 2 * halfwidth]);
            // n == halfheight - 1
            nIndex = (halfheight - 1) * halfwidth + i;
            rIndex = (height - 2) * halfwidth + i;
            if (height == (2 * halfheight))// Yes this can be different if height % 2 != 0
            {
                output[nIndex] = rh0 * input[rIndex] + rh1 * (input[rIndex - halfwidth] + input[rIndex + halfwidth])
                        + rh2 * (input[rIndex - 2 * halfwidth] + input[rIndex + halfwidth]);
            } else {
                output[nIndex] = rh0 * input[rIndex - halfwidth] + rh1 * (input[rIndex - 2 * halfwidth] + input[rIndex])
                        + rh2 * (input[rIndex - 3 * halfwidth] + input[rIndex + halfwidth]);
            }
        }
    }

    private static void convertDoubleToFloat(final double source[], final float target[], final int size) {
        for (int i = 0; i < size; i++) {
            target[i] = (float) source[i];
        }
    }

    private void constructTargetImagePyramid() {
        int width = (int) sharedContext.img.dimension(0);
        int height = (int) sharedContext.img.dimension(1);
        converter.convertTo(scat.targetArray, doubleTargetCoefficients);
        // pre-multiply the image for cubic spline interpolation
        PlainJavaCPUAligner.premultiplyCubicBSpline(doubleTargetCoefficients, width * height);
        // Conversion to B-spline coefficients along X axis
        PlainJavaCPUAligner.cubicBSplinePrefilter2DXhp(doubleTargetCoefficients, width, height);
        // pre-multiply again
        PlainJavaCPUAligner.premultiplyCubicBSpline(doubleTargetCoefficients, width * height);
        // Now along the Y-axis
        PlainJavaCPUAligner.cubicBSplinePrefilter2DYhp(doubleTargetCoefficients, width, height);
        convertDoubleToFloat(doubleTargetCoefficients, targetPyramid[0].Coefficient, width * height);

        // Now prepare the image for resampling by applying the FIR filter of degree 7
        // (out of place mod)
        // X-FIR
        basicToCardinal2DXhpDeg7(targetPyramid[0].Coefficient, entryImageBuffers, width, height);
        // Y-FIR
        basicToCardinal2DYhpDeg7(entryImageBuffers, fullSizedHelperBuffer, width, height);

        // Now start the reduction loop
        // reduce in x direction
        reduceDual1DX(fullSizedHelperBuffer, entryImageBuffers, width, height, width / 2);
        for (int j = 1; j < pyramidDepth; j++) {
            // reduce in y direction
            reduceDual1DY(entryImageBuffers, targetPyramid[j].Coefficient, width / 2, height, height / 2);
            if (j < pyramidDepth - 1) {
                // reduce in x direction
                reduceDual1DX(targetPyramid[j].Coefficient, entryImageBuffers, width / 2, height / 2,
                        (int) (((int) (width / 2)) / 2));// Warning integer division don't change
            }
            width /= 2;
            height /= 2;
            // Now restore the B-spline coefficients
            // pre-multiply
            premultiplyCubicBSplineDeg7(targetPyramid[j].Coefficient, width * height);
            // x-restoration
            cubicBSplinePrefilter2DXhpDeg7(targetPyramid[j].Coefficient, width, height);
            // Don't forget to pre-multiply again
            premultiplyCubicBSplineDeg7(targetPyramid[j].Coefficient, width * height);
            // y-restoration
            cubicBSplinePrefilter2DYhpDeg7(targetPyramid[j].Coefficient, width, height);
        }
    }

    private static void antiSymmetricFirMirrorOffBounds1DXFloat(final float input[], final float output[],
            final int width, final int height) {
        for (int i = 0; i < height; i++) {
            int nIndex = i * width + 1;
            for (int n = 1; n < width - 1; n++, nIndex++) {
                output[nIndex] = (0.5f * (input[nIndex + 1] - input[nIndex - 1]));
            }
            // n == 0
            nIndex = i * width;
            output[nIndex] = (0.5f * (input[nIndex + 1] - input[nIndex]));

            // n == width - 1
            nIndex += width - 1;
            output[nIndex] = (0.5f * (input[nIndex] - input[nIndex - 1]));
        }
    }

    private static void antiSymmetricFirMirrorOffBounds1DYFloat(final float input[], final float output[],
            final int width, final int height) {
        for (int i = 0; i < width; i++) {
            int nIndex = i + width;
            for (int n = 1; n < height - 1; n++, nIndex += width) {
                output[nIndex] = (0.5f * (input[nIndex + width] - input[nIndex - width]));
            }
            // n == 0
            output[i] = (0.5f * (input[i + width] - input[i]));

            // n == width - 1
            nIndex = (height - 1) * width + i;
            output[nIndex] = (0.5f * (input[nIndex] - input[nIndex - width]));
        }
    }

    private static void basicToCardinal2DXhpFloat(final float input[], final float output[], final int width,
            final int height) {
        for (int i = 0; i < height; i++) {
            int nIndex = i * width + 1;
            for (int n = 1; n < width - 1; n++, nIndex++) {
                output[nIndex] = (h0D3 * input[nIndex] + h1D3 * (input[nIndex - 1] + input[nIndex + 1]));
            }
            // n == 0
            nIndex = i * width;
            output[nIndex] = (h0D3 * input[nIndex] + h1D3 * (input[nIndex] + input[nIndex + 1]));
            // n == width - 1
            nIndex += width - 1;
            output[nIndex] = (h0D3 * input[nIndex] + h1D3 * (input[nIndex - 1] + input[nIndex]));
        }
    }

    private static void basicToCardinal2DYhpFloat(final float input[], final float output[], final int width,
            final int height) {
        for (int i = 0; i < width; i++) {
            int nIndex = i + width;
            for (int n = 1; n < height - 1; n++, nIndex += width) {
                output[nIndex] = (h0D3 * input[nIndex] + h1D3 * (input[nIndex - width] + input[nIndex + width]));
            }
            // n == 0
            output[i] = (h0D3 * input[i] + h1D3 * (input[i] + input[i + width]));
            // n == width - 1
            nIndex = (height - 1) * width + i;
            output[nIndex] = (h0D3 * input[nIndex] + h1D3 * (input[nIndex - width] + input[nIndex]));
        }
    }

    private void constructSourceImagePyramid() {
        int width = (int) sharedContext.img.dimension(0);
        int height = (int) sharedContext.img.dimension(1);
        converter.convertTo(scat.sourceArray, doubleSourceImage); // converts to double
        System.arraycopy(doubleSourceImage, 0, doublefullSizedHelperBuffer, 0, width * height);
        convertDoubleToFloat(doublefullSizedHelperBuffer, sourcePyramid[0].Image, width * height);

        // Conversion to B-spline coefficients
        // pre-multiply the image for cubic spline interpolation
        PlainJavaCPUAligner.premultiplyCubicBSpline(doublefullSizedHelperBuffer, width * height);
        // Conversion to B-spline coefficients along X axis
        PlainJavaCPUAligner.cubicBSplinePrefilter2DXhp(doublefullSizedHelperBuffer, width, height);

        // X-derivatives
        PlainJavaCPUAligner.antiSymmetricFirMirrorOffBounds1DX(doublefullSizedHelperBuffer, doubleSourcexGradient, width, height);
        convertDoubleToFloat(doubleSourcexGradient, sourcePyramid[0].xGradient, width * height);

        // pre-multiply again
        PlainJavaCPUAligner.premultiplyCubicBSpline(doublefullSizedHelperBuffer, width * height);
        // Now along Y-axis
        PlainJavaCPUAligner.cubicBSplinePrefilter2DYhp(doublefullSizedHelperBuffer, width, height);
        // for the source only the images in the pyramid are needed, so no need to copy the
        // B-spline coefficients

        // The Y-derivatives from the Y-coefficients still need to be calculated
        // First calculate the Y-coefficients and only the Y-coefficients
        // Has to be pre-multiplied by lambda again
        // To avoid copying data an out-of-place modifying calculation will be used
        PlainJavaCPUAligner.targetedPremultiplyCubicBSpline(doubleSourceImage, doublefullSizedHelperBuffer2, width * height);
        PlainJavaCPUAligner.cubicBSplinePrefilter2DYhp(doublefullSizedHelperBuffer2, width, height);
        // Now calculate the derivatives in Y-direction
        PlainJavaCPUAligner.antiSymmetricFirMirrorOffBounds1DY(doublefullSizedHelperBuffer2, doubleSourceyGradient, width, height);
        convertDoubleToFloat(doubleSourceyGradient, sourcePyramid[0].yGradient, width * height);
        convertDoubleToFloat(doublefullSizedHelperBuffer, entryImageBuffers, width * height);

        // Prepare the image for resampling by applying the FIR filter of degree 7
        // (out of place mod)
        // X-FIR
        basicToCardinal2DXhpDeg7(entryImageBuffers, fullSizedHelperBuffer, width, height);
        // Y-FIR
        basicToCardinal2DYhpDeg7(fullSizedHelperBuffer, entryImageBuffers, width, height);

        // Start the reduction loop
        // reduce in x direction
        reduceDual1DX(entryImageBuffers, fullSizedHelperBuffer, width, height, width / 2);
        for (int j = 1; j < pyramidDepth; j++) {
            // reduce in y direction
            reduceDual1DY(fullSizedHelperBuffer, sourcePyramid[j].Image, width / 2, height, height / 2);
            if (j < pyramidDepth - 1) {
                // reduce in x direction
                reduceDual1DX(sourcePyramid[j].Image, fullSizedHelperBuffer, width / 2, height / 2,
                        (int) (((int) (width / 2)) / 2));// Warning integer division don't change
            }
            width /= 2;
            height /= 2;
            // Restore the B-spline coefficients
            // pre-multiply
            premultiplyCubicBSplineDeg7(sourcePyramid[j].Image, width * height);
            // x-restoration
            cubicBSplinePrefilter2DXhpDeg7(sourcePyramid[j].Image, width, height);
            // pre-multiply again
            premultiplyCubicBSplineDeg7(sourcePyramid[j].Image, width * height);
            // y-restoration
            cubicBSplinePrefilter2DYhpDeg7(sourcePyramid[j].Image, width, height);
            // Now the downsampled images need to be restored and the derivatives have to be
            // calculated
            antiSymmetricFirMirrorOffBounds1DXFloat(sourcePyramid[j].Image, entryImageBuffers, width, height);
            // Because all filters are linearly separable the Y
            // coefficients may simply be restored on the X-diff and vice versa
            basicToCardinal2DYhpFloat(entryImageBuffers, sourcePyramid[j].xGradient, width, height);
            // Now Y
            antiSymmetricFirMirrorOffBounds1DYFloat(sourcePyramid[j].Image, entryImageBuffers, width, height);
            basicToCardinal2DXhpFloat(entryImageBuffers, sourcePyramid[j].yGradient, width, height);

            // Now restore the actual downsampled image from the B-spline coefficients
            // residing in sourcePyramid[j].Image
            basicToCardinal2DXhpFloat(sourcePyramid[j].Image, entryImageBuffers, width, height);
            basicToCardinal2DYhpFloat(entryImageBuffers, sourcePyramid[j].Image, width, height);
        }
    }

    private void doRegistration() {
        iterationPower = (int) Math.pow(2.0, (double) pyramidDepth);
        
        switch(sharedContext.transformationType) {
        case TRANSLATION:
            for (int i = pyramidDepth - 1; i > 0; i--) {
                iterationPower /= 2;
                inverseMarquardtLevenbergTranslationOptimization(i);
                // simply scale up the translation
                offsetx *= 2.0;
                offsety *= 2.0;
            }
            iterationPower /= 2;
            // inverseMarquardtLevenbergTranslationOptimization(0);
            // Unlike the GPU float version the CPU float version is not fast enough to significantly gain
            // performance by doing the final step first in float and then in double precision.
            // The perfect initial guess calculated using float arithmetics can now be used
            // to do a last round with double precision.
            doubleInverseMarquardtLevenbergTranslationOptimization();
            break;
        case RIGIDBODY:
            for (int i = pyramidDepth - 1; i > 0; i--) {
                iterationPower /= 2;
                inverseMarquardtLevenbergRigidBodyOptimization(i);
                // Scale up (but the rotation is not scale dependent so simply scale up the translation)
                offsetx *= 2.0;
                offsety *= 2.0;
            }
            iterationPower /= 2;
            // inverseMarquardtLevenbergRigidBodyOptimization(0);
            // Unlike the GPU float version the CPU float version is not fast enough to significantly gain
            // performance by doing the final step first in float and then in double precision.
            // The perfect initial guess calculated using float arithmetics can now be used
            // to do a last round with double precision.
            doubleInverseMarquardtLevenbergRigidBodyOptimization();
            break;
        case SCALEDROTATION:
        	for(int i = pyramidDepth - 1;i > 0;i--)
            {
                iterationPower /= 2;
                inverseMarquardtLevenbergScaledRotationOptimization(i);
                // scale up (but the rotation and scale are not scale dependent)
                offsetx *= 2.0;
                offsety *= 2.0;
            }
            iterationPower /= 2;
            doubleInverseMarquardtLevenbergScaledRotationOptimization();
            break;
        case AFFINE:
        	for(int i = pyramidDepth - 1;i > 0;i--)
            {
                iterationPower /= 2;
                inverseMarquardtLevenbergAffineOptimization(i);
                // scale up (but the rotation and scale are not scale dependent)
                offsetx *= 2.0;
                offsety *= 2.0;
            }
            iterationPower /= 2;
            doubleInverseMarquardtLevenbergAffineOptimization();
            break;
        }
    }

    private void doubleInverseMarquardtLevenbergTranslationOptimization() {
    	double[] update = { 0.0, 0.0 };
        double bestMeanSquares = 0.0;
        double meanSquares = 0.0;
        double lambda = 1.0;
        double displacement;
        int iteration = 0;
        // first initialize the matrix with the current transformation (upscaling between the steps)
        double currentoffsetx;
        double currentoffsety;
        bestMeanSquares = getTranslationMeanSquaresDouble(offsetx, offsety);
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
            meanSquares = getTranslationMeanSquaresDouble(currentoffsetx, currentoffsety);

            iteration++;
            if (meanSquares < bestMeanSquares) {
                bestMeanSquares = meanSquares;
                lambda /= 4.0;
                offsetx = currentoffsetx;
                offsety = currentoffsety;
            } else {
                lambda *= 4.0;
            }
        } while ((iteration < (10 * iterationPower - 1)) && (0.001 <= displacement));
        StaticUtility.invertGauss(hessian);
        update = StaticUtility.matrixMultiply(hessian, gradient);
        currentoffsetx = offsetx + update[0];
        currentoffsety = offsety + update[1];
        meanSquares = getTranslationMeanSquaresWithoutHessianDouble(currentoffsetx, currentoffsety);
        iteration++;
        if (meanSquares < bestMeanSquares) {
            offsetx = currentoffsetx;
            offsety = currentoffsety;
        }
    }

    private void doubleInverseMarquardtLevenbergRigidBodyOptimization() {
        double[] update = { 0.0, 0.0, 0.0 };
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
        bestMeanSquares = getRigidBodyMeanSquaresDouble(offsetx, offsety, this.angle);
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
            displacement = Math.sqrt(update[1] * update[1] + update[2] * update[2])
                    + 0.25 * Math.sqrt((double) (targetPyramid[0].width * targetPyramid[0].width)
                            + (double) (targetPyramid[0].height * targetPyramid[0].height)) * Math.abs(update[0]);
            c = Math.cos(update[0]);
            s = Math.sin(update[0]);
            currentoffsetx = (offsetx + update[1]) * c - (offsety + update[2]) * s;
            currentoffsety = (offsetx + update[1]) * s + (offsety + update[2]) * c;
            meanSquares = getRigidBodyMeanSquaresDouble(currentoffsetx, currentoffsety, currentangle);

            iteration++;
            if (meanSquares < bestMeanSquares) {
                bestMeanSquares = meanSquares;
                lambda /= 4.0;
                offsetx = currentoffsetx;
                offsety = currentoffsety;
                this.angle = currentangle;
            } else {
                lambda *= 4.0;
            }
        } while ((iteration < (10 * iterationPower - 1)) && (0.001 <= displacement));
        StaticUtility.invertGauss(hessian);
        update = StaticUtility.matrixMultiply(hessian, gradient);
        currentangle = this.angle - update[0];
        c = Math.cos(update[0]);
        s = Math.sin(update[0]);
        currentoffsetx = (offsetx + update[1]) * c - (offsety + update[2]) * s;
        currentoffsety = (offsetx + update[1]) * s + (offsety + update[2]) * c;
        meanSquares = getRigidBodyMeanSquaresWithoutHessianDouble(currentoffsetx, currentoffsety, currentangle);
        iteration++;
        if (meanSquares < bestMeanSquares) {
            offsetx = currentoffsetx;
            offsety = currentoffsety;
            this.angle = currentangle;
        }
    }
    
    private void doubleInverseMarquardtLevenbergScaledRotationOptimization() {
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
        bestMeanSquares = getScaledRotationMeanSquaresDouble(offsetx,offsety,this.angle,this.scale);
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
            meanSquares = getScaledRotationMeanSquaresDouble(currentoffsetx,currentoffsety,currentangle,currentscale);

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
        meanSquares = getScaledRotationMeanSquaresWithoutHessianDouble(currentoffsetx,currentoffsety,currentangle,currentscale);
        iteration++;
        if (meanSquares < bestMeanSquares) {
            offsetx = currentoffsetx;
            offsety = currentoffsety;
            this.angle = currentangle;
            this.scale = currentscale;
        }
    }
    
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
        bestMeanSquares = getAffineMeanSquaresDouble(offsetx, offsety, a11, a12, a21, a22);
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
                                  + (double)(targetPyramid[0].height * targetPyramid[0].height));
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
            meanSquares = getAffineMeanSquaresDouble(currentoffsetx, currentoffsety,
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

        meanSquares = getAffineMeanSquaresWithoutHessianDouble(currentoffsetx, currentoffsety,
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

    private void inverseMarquardtLevenbergTranslationOptimization(int pyramidIndex) {
        double[] update = { 0.0, 0.0 };
        double bestMeanSquares = 0.0;
        double meanSquares = 0.0;
        double lambda = 1.0;
        double displacement;
        int iteration = 0;
        // first initialize the matrix with the current transformation (upscaling between the steps)
        double currentoffsetx;
        double currentoffsety;
        bestMeanSquares = getTranslationMeanSquares(pyramidIndex, (float) offsetx, (float) offsety);
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
            meanSquares = getTranslationMeanSquares(pyramidIndex, (float) currentoffsetx, (float) currentoffsety);

            iteration++;
            if (meanSquares < bestMeanSquares) {
                bestMeanSquares = meanSquares;
                lambda /= 4.0;
                offsetx = currentoffsetx;
                offsety = currentoffsety;
            } else {
                lambda *= 4.0;
            }
        } while ((iteration < (10 * iterationPower - 1)) && (0.001 <= displacement));
        StaticUtility.invertGauss(hessian);
        update = StaticUtility.matrixMultiply(hessian, gradient);
        
        currentoffsetx = offsetx + update[0];
        currentoffsety = offsety + update[1];
        meanSquares = getTranslationMeanSquaresWithoutHessian(pyramidIndex, (float) currentoffsetx, (float) currentoffsety);
        iteration++;
        if (meanSquares < bestMeanSquares) {
            offsetx = currentoffsetx;
            offsety = currentoffsety;
        }
    }
    
    private void inverseMarquardtLevenbergRigidBodyOptimization(int pyramidIndex) {
        double[] update = { 0.0, 0.0, 0.0 };
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
        bestMeanSquares = getRigidBodyMeanSquares(pyramidIndex, (float) offsetx, (float) offsety, (float) this.angle);
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
            displacement = Math.sqrt(update[1] * update[1] + update[2] * update[2]) + 0.25
                    * Math.sqrt((double) (targetPyramid[pyramidIndex].width * targetPyramid[pyramidIndex].width)
                            + (double) (targetPyramid[pyramidIndex].height * targetPyramid[pyramidIndex].height))
                    * Math.abs(update[0]);
            c = Math.cos(update[0]);
            s = Math.sin(update[0]);
            currentoffsetx = (offsetx + update[1]) * c - (offsety + update[2]) * s;
            currentoffsety = (offsetx + update[1]) * s + (offsety + update[2]) * c;
            meanSquares = getRigidBodyMeanSquares(pyramidIndex, (float) currentoffsetx, (float) currentoffsety,
                    (float) currentangle);

            iteration++;
            if (meanSquares < bestMeanSquares) {
                bestMeanSquares = meanSquares;
                lambda /= 4.0;
                offsetx = currentoffsetx;
                offsety = currentoffsety;
                this.angle = currentangle;
            } else {
                lambda *= 4.0;
            }
        } while ((iteration < (10 * iterationPower - 1)) && (0.001 <= displacement));
        StaticUtility.invertGauss(hessian);
        update = StaticUtility.matrixMultiply(hessian, gradient);
        currentangle = this.angle - update[0];
        c = Math.cos(update[0]);
        s = Math.sin(update[0]);
        currentoffsetx = (offsetx + update[1]) * c - (offsety + update[2]) * s;
        currentoffsety = (offsetx + update[1]) * s + (offsety + update[2]) * c;
        meanSquares = getRigidBodyMeanSquaresWithoutHessian(pyramidIndex, (float) currentoffsetx,
                (float) currentoffsety, (float) currentangle);
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
        bestMeanSquares = getScaledRotationMeanSquares(pyramidIndex,(float)offsetx,(float)offsety,(float)this.angle,(float)this.scale);
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
            meanSquares = getScaledRotationMeanSquares(pyramidIndex,(float)currentoffsetx,(float)currentoffsety,(float)currentangle,(float)currentscale);

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
        meanSquares = getScaledRotationMeanSquaresWithoutHessian(pyramidIndex,(float)currentoffsetx,(float)currentoffsety,(float)currentangle,(float)currentscale);
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
        bestMeanSquares = getAffineMeanSquares(pyramidIndex, (float)offsetx, (float)offsety, (float)a11, (float)a12, (float)a21, (float)a22);
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
            meanSquares = getAffineMeanSquares(pyramidIndex, (float)currentoffsetx, (float)currentoffsety,
                                               (float)currenta11, (float)currenta12, (float)currenta21, (float)currenta22);

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

        meanSquares = getAffineMeanSquaresWithoutHessian(pyramidIndex, (float)currentoffsetx, (float)currentoffsety,
                                                          (float)currenta11, (float)currenta12, (float)currenta21, (float)currenta22);
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
    
    private double getTranslationMeanSquaresDouble(double currentoffsetx, double currentoffsety) {
        // First reset the global values which will not be reset in the loop
    	Arrays.fill(gradient, 0.0);
        for (int i = 0; (i < 2); i++) {
            Arrays.fill(hessian[i], 0.0);
        }
        
        final int width = (int) sharedContext.img.dimension(0);
        final int height = (int) sharedContext.img.dimension(1);
        final int doubletargetwidth = width * 2;
        final int doubletargetheight = height * 2;
        int nIndex = 0;
        int area = 0;
        double s;
        double msqe = 0.0;// mean square error
        double coordx;
        double rescoordx;
        double coordy;
        double rescoordy;
        int mskx;
        int msky;
        for (int i = 0; i < height; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx;
            coordy = currentoffsety + ((double) i);
            for (int n = 0; n < width; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < width) && (msky >= 0) && (msky < height)) {
                    area++;
                    PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubletargetwidth, width, xInterpolationIndices);
                    PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubletargetheight, height, width, yInterpolationIndices);

                    rescoordx = PlainJavaCPUAligner.getFractional(coordx);
                    rescoordy = PlainJavaCPUAligner.getFractional(coordy);
                    
                    s = PlainJavaCPUAligner.interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, doubleTargetCoefficients);

                    // now calculate the return value
                    double diff = doubleSourceImage[nIndex] - s;
                    msqe += diff * diff;
                    /*
                     * TODO/FIXME/KNOWN ISSUE: The following summation is MUCH worse than the
                     * parallel sum reduction done on the GPU, because (relatively speaking) small
                     * numbers are added to an ever growing larger number reducing the precision in
                     * the outcome. Currently I ignore this like the original implementation, but
                     * this is one of many reasons why the GPU version and the CPU version will
                     * never yield the same results!
                     */
                    gradient[0] += diff * doubleSourcexGradient[nIndex];
                    gradient[1] += diff * doubleSourceyGradient[nIndex];
                    hessian[0][0] += doubleSourcexGradient[nIndex] * doubleSourcexGradient[nIndex];
                    hessian[0][1] += doubleSourcexGradient[nIndex] * doubleSourceyGradient[nIndex];
                    hessian[1][1] += doubleSourceyGradient[nIndex] * doubleSourceyGradient[nIndex];
                }
                // walk along the X-vector direction
                coordx += 1.0;
            }
        }
        // symmetrize hessian
        for (int i = 1; (i < 2); i++) {
            for (int j = 0; (j < i); j++) {
                hessian[i][j] = hessian[j][i];
            }
        }
        return msqe / ((double) area);
    }

    private double getRigidBodyMeanSquaresDouble(double currentoffsetx, double currentoffsety, double currentangle) {
        // First reset the global values which will not be reset in the loop
        Arrays.fill(gradient, 0.0);
        for (int i = 0; (i < 3); i++) {
            Arrays.fill(hessian[i], 0.0);
        }
        
        final int width = (int) sharedContext.img.dimension(0);
        final int height = (int) sharedContext.img.dimension(1);
        final int doubletargetwidth = width * 2;
        final int doubletargetheight = height * 2;
        int nIndex = 0;
        int area = 0;
        double s;
        double msqe = 0.0;// mean square error
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
        for (int i = 0; i < height; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx + ((double) i) * yvecx;
            coordy = currentoffsety + ((double) i) * yvecy;
            for (int n = 0; n < width; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < width) && (msky >= 0) && (msky < height)) {
                    area++;

                    PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubletargetwidth, width, xInterpolationIndices);
                    PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubletargetheight, height, width, yInterpolationIndices);

                    rescoordx = PlainJavaCPUAligner.getFractional(coordx);
                    rescoordy = PlainJavaCPUAligner.getFractional(coordy);
                    
                    s = PlainJavaCPUAligner.interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, doubleTargetCoefficients);
                    
                    // calculate the values for returning
                    double diff = doubleSourceImage[nIndex] - s;
                    msqe += diff * diff;
                    double theta = doubleSourceyGradient[nIndex] * (double) n - doubleSourcexGradient[nIndex] * (double) i; 
                    /*
                     * TODO/FIXME/KNOWN ISSUE: The following summation is MUCH worse than the
                     * parallel sum reduction done on the GPU, because (relatively speaking) small
                     * numbers are added to an ever growing larger number reducing the precision in
                     * the outcome. Currently I ignore this like the original implementation, but
                     * this is one of many reasons why the GPU version and the CPU version will
                     * never yield the same results!
                     */
                    gradient[0] += diff * theta;
                    gradient[1] += diff * doubleSourcexGradient[nIndex];
                    gradient[2] += diff * doubleSourceyGradient[nIndex];
                    hessian[0][0] += theta * theta;
                    hessian[0][1] += theta * doubleSourcexGradient[nIndex];
                    hessian[0][2] += theta * doubleSourceyGradient[nIndex];
                    hessian[1][1] += doubleSourcexGradient[nIndex] * doubleSourcexGradient[nIndex];
                    hessian[1][2] += doubleSourcexGradient[nIndex] * doubleSourceyGradient[nIndex];
                    hessian[2][2] += doubleSourceyGradient[nIndex] * doubleSourceyGradient[nIndex];
                }
                // walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
        // symmetrize hessian
        for (int i = 1; (i < 3); i++) {
            for (int j = 0; (j < i); j++) {
                hessian[i][j] = hessian[j][i];
            }
        }
        return msqe / ((double) area);
    }
    
    private double getScaledRotationMeanSquaresDouble(double currentoffsetx, double currentoffsety, double currentangle, double currentscale) {
        // First reset the global values which will not be reset in the loop
        Arrays.fill(gradient, 0.0);
        for (int i = 0; (i < 4); i++) {
            Arrays.fill(hessian[i], 0.0);
        }
        
        final int width = (int) sharedContext.img.dimension(0);
        final int height = (int) sharedContext.img.dimension(1);
        final int doubletargetwidth = width * 2;
        final int doubletargetheight = height * 2;
        int nIndex = 0;
        int area = 0;
        double s;
        double msqe = 0.0;// mean square error
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
        for (int i = 0; i < height; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx + ((double) i) * yvecx;
            coordy = currentoffsety + ((double) i) * yvecy;
            for (int n = 0; n < width; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < width) && (msky >= 0) && (msky < height)) {
                    area++;

                    PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubletargetwidth, width, xInterpolationIndices);
                    PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubletargetheight, height, width, yInterpolationIndices);

                    rescoordx = PlainJavaCPUAligner.getFractional(coordx);
                    rescoordy = PlainJavaCPUAligner.getFractional(coordy);
                    
                    s = PlainJavaCPUAligner.interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, doubleTargetCoefficients);
                    
                    // calculate the return values
                    double diff = doubleSourceImage[nIndex] - s;
                    msqe += diff * diff;
                    double theta = doubleSourceyGradient[nIndex] * (double)n - doubleSourcexGradient[nIndex] * (double)i;
                    double j_scale = (((double)n) * doubleSourcexGradient[nIndex] + ((double)i) * doubleSourceyGradient[nIndex]); // scale contribution to j
                    /*
                    TODO/FIXME/KNOWN ISSUE:
                    The following summation is MUCH worse than the parallel sum reduction done on the GPU, because (relatively speaking)
                    small numbers are added to an ever growing larger number, reducing the precision in the outcome. Currently
                    I ignore this like the original implementation, but this is one of many reasons why the GPU version and the CPU version
                    will never yield the same results!
                    */
                    gradient[0] += diff * j_scale;
                    gradient[1] += diff * theta;
                    gradient[2] += diff * doubleSourcexGradient[nIndex];
                    gradient[3] += diff * doubleSourceyGradient[nIndex];
                    hessian[0][0] += j_scale * j_scale;
                    hessian[0][1] += j_scale * theta;
                    hessian[0][2] += j_scale * doubleSourcexGradient[nIndex];
                    hessian[0][3] += j_scale * doubleSourceyGradient[nIndex];
                    hessian[1][1] += theta * theta;
                    hessian[1][2] += theta * doubleSourcexGradient[nIndex];
                    hessian[1][3] += theta * doubleSourceyGradient[nIndex];
                    hessian[2][2] += doubleSourcexGradient[nIndex] * doubleSourcexGradient[nIndex];
                    hessian[2][3] += doubleSourcexGradient[nIndex] * doubleSourceyGradient[nIndex];
                    hessian[3][3] += doubleSourceyGradient[nIndex] * doubleSourceyGradient[nIndex];
                }
                // walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
        // symmetrize hessian
        for (int i = 1; (i < 4); i++) {
            for (int j = 0; (j < i); j++) {
                hessian[i][j] = hessian[j][i];
            }
        }
        return msqe / ((double) area);
    }
    
    
    private double getAffineMeanSquaresDouble(double currentoffsetx, double currentoffsety,
            double currenta11, double currenta12,
            double currenta21, double currenta22) {
        // First reset the global values which will not be reset in the loop
    	Arrays.fill(gradient, 0.0);
        for (int i = 0; i < 6; i++) {
            Arrays.fill(hessian[i], 0.0);
        }
        //WARNING: Unlike the PlainJavaCPUAligner the width and height of the source and target MUST be the same
        final int width = (int) sharedContext.img.dimension(0);
        final int height = (int) sharedContext.img.dimension(1);
        final int doubletargetwidth = width * 2;
        final int doubletargetheight = height * 2;
        int nIndex = 0;
        int area = 0;
        double s;
        double msqe = 0.0;// mean square error
        double xvecx = currenta11;
        double xvecy = currenta21;
        double yvecx = currenta12;
        double yvecy = currenta22;
        double coordx;
        double rescoordx;
        double coordy;
        double rescoordy;
        int mskx;
        int msky;
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
        for (int i = 0; i < height; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx + ((double) i) * yvecx;
            coordy = currentoffsety + ((double) i) * yvecy;
            for (int n = 0; n < width; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < width) && (msky >= 0) && (msky < height)) {
                    area++;
                    
                    PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubletargetwidth, width, xInterpolationIndices);
                    PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubletargetheight, height, width, yInterpolationIndices);

                    rescoordx = PlainJavaCPUAligner.getFractional(coordx);
                    rescoordy = PlainJavaCPUAligner.getFractional(coordy);
                    
                    s = PlainJavaCPUAligner.interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, doubleTargetCoefficients);

                    // ---- Gradient and Hessian accumulation ----

                    /*
                     * Residual: r_i = f(x_i) - g(T_p(x_i))
                     * where f is the source image and g is the B-spline-interpolated target.
                     * We reuse rescoordx to store the residual (same as translation/rigid body).
                     */
                    double diff = doubleSourceImage[nIndex] - s;
                    msqe += diff * diff;
                    
                    /*
                    TODO/FIXME/KNOWN ISSUE:
                    The following summation is MUCH worse than the parallel sum reduction done on the GPU, because (relatively speaking)
                    small numbers are added to an ever growing larger number, reducing the precision in the outcome. Currently
                    I ignore this like the original implementation, but this is one of many reasons why the GPU version and the CPU version
                    will never yield the same results!
                    */
                    
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
                    dx = doubleSourcexGradient[nIndex];
                    dy = doubleSourceyGradient[nIndex];
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
        // symmetrize hessian
        for (int i = 1; (i < 6); i++) {
            for (int j = 0; (j < i); j++) {
                hessian[i][j] = hessian[j][i];
            }
        }
        return msqe / ((double) area);
    }
    
    
    
    
    
    
    
    
    
    
    /**
     * Computes 4 mirror-boundary B-spline X-interpolation indices.
     * Fills indices[c] with the mirrored column index (no stride multiplication).
     *
     * @param coord           the (possibly fractional) x-coordinate
     * @param doubletargetwidth   2 * targetwidth
     * @param targetwidth     width of the target image
     * @param indices         output array to fill (length >= 4)
     */
    static void computeXInterpolationIndicesFloat(float coord, int doubletargetwidth, int targetwidth, int[] indices)
    {
        int p = (coord >= 0) ? (((int)coord) + 2) : (((int)coord) + 1);
        for (int c = 0; c < 4; c++, p--)
        {
            int q = (p < 0) ? (-1 - p) : p;
            q = (q < doubletargetwidth) ? q : q % doubletargetwidth;
            indices[c] = q >= targetwidth ? (doubletargetwidth - 1 - q) : q;
        }
    }

    /**
     * Computes 4 mirror-boundary B-spline Y-interpolation indices,
     * linearized by multiplying by targetwidth for direct use as row offsets.
     *
     * @param coord              the (possibly fractional) y-coordinate
     * @param doubletargetheight 2 * targetheight
     * @param targetheight       height of the target image
     * @param targetwidth        width of the target image (stride multiplier)
     * @param indices            output array to fill (length >= 4)
     */
    static void computeYInterpolationIndicesFloat(float coord, int doubletargetheight, int targetheight, int targetwidth, int[] indices)
    {
        int p = (coord >= 0) ? (((int)coord) + 2) : (((int)coord) + 1);
        for (int c = 0; c < 4; c++, p--)
        {
            int q = (p < 0) ? (-1 - p) : p;
            q = (q < doubletargetheight) ? q : q % doubletargetheight;
            indices[c] = q >= targetheight ? (doubletargetheight - 1 - q) * targetwidth : q * targetwidth;
        }
    }


    /**
     * Returns the fractional (sub-pixel) part of a coordinate.
     * Handles negative coordinates correctly (floor, not truncation).
     */
    static float getFractionalFloat(float coord)
    {
        return coord - (coord >= 0.0 ? ((float)((int)coord)) : ((float)(((int)coord) - 1)));
    }
    
    /**
     * Computes the cubic B-spline interpolated value at the given fractional
     * coordinates, using the precomputed interpolation indices.
     *
     * This is mathematically equivalent to:
     *   - Computing xWeights[0..3] and yWeights[0..3] from rescoordx and rescoordy
     *   - Performing the separable 4x4 B-spline interpolation over target[]
     *
     * The cubic B-spline basis functions used here are the standard ones:
     *   w3 = (1 - t)^3 / 6
     *   w2 = 2/3 - t^2*(2-t)/2
     *   w0 = t^2*t / 6
     *   w1 = 1 - w0 - w2 - w3
     *
     * @param rescoordx            fractional x-coordinate in [0, 1)
     * @param rescoordy            fractional y-coordinate in [0, 1)
     * @param xInterpolationIndices precomputed mirror-boundary column indices (length 4)
     * @param yInterpolationIndices precomputed mirror-boundary row offsets   (length 4)
     * @param target               the B-spline coefficient array
     * @return                     the interpolated value at (rescoordx, rescoordy)
     */
    static float interpolateCubicBSplineFloat(
            float rescoordx, float rescoordy,
            int[] xInterpolationIndices, int[] yInterpolationIndices,
            float[] target)
    {
        // --- X weights ---
    	float sx = 1.0f - rescoordx;
    	float xw3 = (sx * sx * sx) / 6.0f;
    	float sx2 = rescoordx * rescoordx;
    	float xw2 = (2.0f / 3.0f) - 0.5f * sx2 * (2.0f - rescoordx);
    	float xw0 = sx2 * rescoordx / 6.0f;
    	float xw1 = 1.0f - xw0 - xw2 - xw3;

        // --- Y weights ---
        float sy = 1.0f - rescoordy;
        float yw3 = (sy * sy * sy) / 6.0f;
        float sy2 = rescoordy * rescoordy;
        float yw2 = (2.0f / 3.0f) - 0.5f * sy2 * (2.0f - rescoordy);
        float yw0 = sy2 * rescoordy / 6.0f;
        float yw1 = 1.0f - yw0 - yw2 - yw3;

        float[] xWeightsLocal = { xw0, xw1, xw2, xw3 };
        float[] yWeightsLocal = { yw0, yw1, yw2, yw3 };

        // --- Separable 4x4 interpolation ---
        float s = 0.0f;
        for (int y = 0; y < 4; y++)
        {
            int tmpindex = yInterpolationIndices[y];
            float row = 0.0f;
            for (int x = 0; x < 4; x++)
            {
                row += xWeightsLocal[x] * target[tmpindex + xInterpolationIndices[x]];
            }
            s += yWeightsLocal[y] * row;
        }
        return s;
    }
    
    private double getTranslationMeanSquares(int pyramidIndex, float currentoffsetx, float currentoffsety) {
        // First reset the global values which will not be reset in the loop
    	Arrays.fill(gradient, 0.0);
        for (int i = 0; (i < 2); i++) {
            Arrays.fill(hessian[i], 0.0);
        }
        
        Arrays.fill(fgradient, 0.0f);
        for (int i = 0; (i < 2); i++) {
            Arrays.fill(fhessian[i], 0.0f);
        }

        final int width = (int) sourcePyramid[pyramidIndex].width;
        final int height = (int) sourcePyramid[pyramidIndex].height;
        final float[] source = sourcePyramid[pyramidIndex].Image;
        final float[] xGradient = sourcePyramid[pyramidIndex].xGradient;
        final float[] yGradient = sourcePyramid[pyramidIndex].yGradient;
        final int targetwidth = (int) targetPyramid[pyramidIndex].width;
        final int doubletargetwidth = targetwidth * 2;
        final int targetheight = (int) targetPyramid[pyramidIndex].height;
        final int doubletargetheight = targetheight * 2;
        final float[] target = targetPyramid[pyramidIndex].Coefficient;
        int nIndex = 0;
        int area = 0;
        float s;
        float msqe = 0.0f;// mean square error
        float coordx;
        float rescoordx;
        float coordy;
        float rescoordy;
        int mskx;
        int msky;
        for (int i = 0; i < height; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx;
            coordy = currentoffsety + ((float) i);
            for (int n = 0; n < width; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < targetwidth) && (msky >= 0) && (msky < targetheight)) {
                    area++;
                    computeXInterpolationIndicesFloat(coordx, doubletargetwidth, targetwidth, xInterpolationIndices);
                    computeYInterpolationIndicesFloat(coordy, doubletargetheight, targetheight, targetwidth, yInterpolationIndices);

                    rescoordx = getFractionalFloat(coordx);
                    rescoordy = getFractionalFloat(coordy);
                    
                    s = interpolateCubicBSplineFloat(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, target);

                    // now calculate the return value
                    float diff = source[nIndex] - s;
                    msqe += diff * diff;
                    /*
                     * TODO/FIXME/KNOWN ISSUE: The following summation is MUCH worse than the
                     * parallel sum reduction done on the GPU, because (relatively speaking) small
                     * numbers are added to an ever growing larger number reducing the precision in
                     * the outcome. Currently I ignore this like the original implementation, but
                     * this is one of many reasons why the GPU version and the CPU version will
                     * never yield the same results!
                     */
                    fgradient[0] += diff * xGradient[nIndex];
                    fgradient[1] += diff * yGradient[nIndex];
                    fhessian[0][0] += xGradient[nIndex] * xGradient[nIndex];
                    fhessian[0][1] += xGradient[nIndex] * yGradient[nIndex];
                    fhessian[1][1] += yGradient[nIndex] * yGradient[nIndex];
                }
                // walk along the X-vector direction
                coordx += 1.0f;
            }
        }
        gradient[0] = fgradient[0];
        gradient[1] = fgradient[1];
        hessian[0][0] = fhessian[0][0];
        hessian[0][1] = fhessian[0][1];
        hessian[1][0] = fhessian[1][0];
        hessian[1][1] = fhessian[1][1];
        // symmetrize hessian
        for (int i = 1; (i < 2); i++) {
            for (int j = 0; (j < i); j++) {
                hessian[i][j] = hessian[j][i];
            }
        }
        return msqe / ((double) area);
    }

    private double getRigidBodyMeanSquares(int pyramidIndex, float currentoffsetx, float currentoffsety,
            float currentangle) {
        // First reset the global values which will not be reset in the loop
        Arrays.fill(gradient, 0.0);
        for (int i = 0; (i < 3); i++) {
            Arrays.fill(hessian[i], 0.0);
        }
        Arrays.fill(fgradient, 0.0f);
        for (int i = 0; (i < 3); i++) {
            Arrays.fill(fhessian[i], 0.0f);
        }
        final int width = (int) sourcePyramid[pyramidIndex].width;
        final int height = (int) sourcePyramid[pyramidIndex].height;
        final float[] source = sourcePyramid[pyramidIndex].Image;
        final float[] xGradient = sourcePyramid[pyramidIndex].xGradient;
        final float[] yGradient = sourcePyramid[pyramidIndex].yGradient;
        final int targetwidth = (int) targetPyramid[pyramidIndex].width;
        final int doubletargetwidth = targetwidth * 2;
        final int targetheight = (int) targetPyramid[pyramidIndex].height;
        final int doubletargetheight = targetheight * 2;
        final float[] target = targetPyramid[pyramidIndex].Coefficient;
        int nIndex = 0;
        int area = 0;
        float s;
        float msqe = 0.0f;// mean square error
        float xvecx = (float) Math.cos(currentangle);
        float xvecy = (float) -Math.sin(currentangle);
        float yvecx = -xvecy;
        float yvecy = xvecx;
        float coordx;
        float rescoordx;
        float coordy;
        float rescoordy;
        int mskx;
        int msky;
        for (int i = 0; i < height; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx + ((float) i) * yvecx;
            coordy = currentoffsety + ((float) i) * yvecy;
            for (int n = 0; n < width; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < targetwidth) && (msky >= 0) && (msky < targetheight)) {
                    area++;

                    computeXInterpolationIndicesFloat(coordx, doubletargetwidth, targetwidth, xInterpolationIndices);
                    computeYInterpolationIndicesFloat(coordy, doubletargetheight, targetheight, targetwidth, yInterpolationIndices);

                    rescoordx = getFractionalFloat(coordx);
                    rescoordy = getFractionalFloat(coordy);
                    
                    s = interpolateCubicBSplineFloat(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, target);

                    // calculate the return values
                    float diff = source[nIndex] - s;
                    msqe += diff * diff;
                    float theta = yGradient[nIndex] * (float)n - xGradient[nIndex] * (float)i;
                    
                    /*
                     * TODO/FIXME/KNOWN ISSUE: The following summation is MUCH worse than the
                     * parallel sum reduction done on the GPU, because (relatively speaking) small
                     * numbers are added to an ever growing larger number reducing the precision in
                     * the outcome. Currently I ignore this like the original implementation, but
                     * this is one of many reasons why the GPU version and the CPU version will
                     * never yield the same results!
                     */
                    fgradient[0] += diff * theta;
                    fgradient[1] += diff * xGradient[nIndex];
                    fgradient[2] += diff * yGradient[nIndex];
                    fhessian[0][0] += theta * theta;
                    fhessian[0][1] += theta * xGradient[nIndex];
                    fhessian[0][2] += theta * yGradient[nIndex];
                    fhessian[1][1] += xGradient[nIndex] * xGradient[nIndex];
                    fhessian[1][2] += xGradient[nIndex] * yGradient[nIndex];
                    fhessian[2][2] += yGradient[nIndex] * yGradient[nIndex];
                }
                // walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
        gradient[0] = fgradient[0];
        gradient[1] = fgradient[1];
        gradient[2] = fgradient[2];
        hessian[0][0] = fhessian[0][0];
        hessian[0][1] = fhessian[0][1];
        hessian[0][2] = fhessian[0][2];
        hessian[1][1] = fhessian[1][1];
        hessian[1][2] = fhessian[1][2];
        hessian[2][2] = fhessian[2][2];
        // symmetrize hessian
        for (int i = 1; (i < 3); i++) {
            for (int j = 0; (j < i); j++) {
                hessian[i][j] = hessian[j][i];
            }
        }
        return msqe / ((double) area);
    }
    
    private double getScaledRotationMeanSquares(int pyramidIndex, float currentoffsetx, float currentoffsety,
            float currentangle, float currentscale) {
        // First reset the global values which will not be reset in the loop
        Arrays.fill(gradient, 0.0);
        for (int i = 0; (i < 4); i++) {
            Arrays.fill(hessian[i], 0.0);
        }
        Arrays.fill(fgradient, 0.0f);
        for (int i = 0; (i < 4); i++) {
            Arrays.fill(fhessian[i], 0.0f);
        }
        final int width = (int) sourcePyramid[pyramidIndex].width;
        final int height = (int) sourcePyramid[pyramidIndex].height;
        final float[] source = sourcePyramid[pyramidIndex].Image;
        final float[] xGradient = sourcePyramid[pyramidIndex].xGradient;
        final float[] yGradient = sourcePyramid[pyramidIndex].yGradient;
        final int targetwidth = (int) targetPyramid[pyramidIndex].width;
        final int doubletargetwidth = targetwidth * 2;
        final int targetheight = (int) targetPyramid[pyramidIndex].height;
        final int doubletargetheight = targetheight * 2;
        final float[] target = targetPyramid[pyramidIndex].Coefficient;
        int nIndex = 0;
        int area = 0;
        float s;
        float msqe = 0.0f;// mean square error
        float xvecx = (float)Math.cos(currentangle) * currentscale;
        float xvecy = (float)-Math.sin(currentangle) * currentscale;
        float yvecx = -xvecy;
        float yvecy = xvecx;
        float coordx;
        float rescoordx;
        float coordy;
        float rescoordy;
        int mskx;
        int msky;
        for (int i = 0; i < height; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx + ((float) i) * yvecx;
            coordy = currentoffsety + ((float) i) * yvecy;
            for (int n = 0; n < width; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < targetwidth) && (msky >= 0) && (msky < targetheight)) {
                    area++;

                    computeXInterpolationIndicesFloat(coordx, doubletargetwidth, targetwidth, xInterpolationIndices);
                    computeYInterpolationIndicesFloat(coordy, doubletargetheight, targetheight, targetwidth, yInterpolationIndices);

                    rescoordx = getFractionalFloat(coordx);
                    rescoordy = getFractionalFloat(coordy);
                    
                    s = interpolateCubicBSplineFloat(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, target);

                    // calculate the return values
                    float diff = source[nIndex] - s;
                    msqe += diff * diff;
                    float theta = yGradient[nIndex] * (float)n - xGradient[nIndex] * (float)i;
                    float j_scale = (((float)n) * xGradient[nIndex] + ((float)i) * yGradient[nIndex]); // scale contribution to j
                    
                    /*
                     * TODO/FIXME/KNOWN ISSUE: The following summation is MUCH worse than the
                     * parallel sum reduction done on the GPU, because (relatively speaking) small
                     * numbers are added to an ever growing larger number reducing the precision in
                     * the outcome. Currently I ignore this like the original implementation, but
                     * this is one of many reasons why the GPU version and the CPU version will
                     * never yield the same results!
                     */
                    fgradient[0] += diff * j_scale;
                    fgradient[1] += diff * theta;
                    fgradient[2] += diff * xGradient[nIndex];
                    fgradient[3] += diff * yGradient[nIndex];
                    fhessian[0][0] += j_scale * j_scale;
                    fhessian[0][1] += j_scale * theta;
                    fhessian[0][2] += j_scale * xGradient[nIndex];
                    fhessian[0][3] += j_scale * yGradient[nIndex];
                    fhessian[1][1] += theta * theta;
                    fhessian[1][2] += theta * xGradient[nIndex];
                    fhessian[1][3] += theta * yGradient[nIndex];
                    fhessian[2][2] += xGradient[nIndex] * xGradient[nIndex];
                    fhessian[2][3] += xGradient[nIndex] * yGradient[nIndex];
                    fhessian[3][3] += yGradient[nIndex] * yGradient[nIndex];
                }
                // walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
        gradient[0] = fgradient[0];
        gradient[1] = fgradient[1];
        gradient[2] = fgradient[2];
        gradient[3] = fgradient[3];
        hessian[0][0] = fhessian[0][0];
        hessian[0][1] = fhessian[0][1];
        hessian[0][2] = fhessian[0][2];
        hessian[0][3] = fhessian[0][3];
        hessian[1][1] = fhessian[1][1];
        hessian[1][2] = fhessian[1][2];
        hessian[1][3] = fhessian[1][3];
        hessian[2][2] = fhessian[2][2];
        hessian[2][3] = fhessian[2][3];
        // symmetrize hessian
        for (int i = 1; (i < 4); i++) {
            for (int j = 0; (j < i); j++) {
                hessian[i][j] = hessian[j][i];
            }
        }
        return msqe / ((double) area);
    }
    
    private double getAffineMeanSquares(int pyramidIndex, float currentoffsetx, float currentoffsety,
    		float currenta11, float currenta12,
    		float currenta21, float currenta22) {
        // First reset the global values which will not be reset in the loop
        Arrays.fill(gradient, 0.0);
        for (int i = 0; (i < 6); i++) {
            Arrays.fill(hessian[i], 0.0);
        }
        Arrays.fill(fgradient, 0.0f);
        for (int i = 0; (i < 6); i++) {
            Arrays.fill(fhessian[i], 0.0f);
        }
        final int width = (int) sourcePyramid[pyramidIndex].width;
        final int height = (int) sourcePyramid[pyramidIndex].height;
        final float[] source = sourcePyramid[pyramidIndex].Image;
        final float[] xGradient = sourcePyramid[pyramidIndex].xGradient;
        final float[] yGradient = sourcePyramid[pyramidIndex].yGradient;
        final int targetwidth = (int) targetPyramid[pyramidIndex].width;
        final int doubletargetwidth = targetwidth * 2;
        final int targetheight = (int) targetPyramid[pyramidIndex].height;
        final int doubletargetheight = targetheight * 2;
        final float[] target = targetPyramid[pyramidIndex].Coefficient;
        int nIndex = 0;
        int area = 0;
        float s;
        float msqe = 0.0f;// mean square error
        float xvecx = currenta11;
        float xvecy = currenta21;
        float yvecx = currenta12;
        float yvecy = currenta22;
        float coordx;
        float rescoordx;
        float coordy;
        float rescoordy;
        int mskx;
        int msky;
        /*
         * Jacobian column variables (recomputed per pixel):
         *   dx0 = n * ∂f/∂x   (∂f/∂a11)
         *   dx1 = i * ∂f/∂x   (∂f/∂a12)
         *   dy0 = n * ∂f/∂y   (∂f/∂a21)
         *   dy1 = i * ∂f/∂y   (∂f/∂a22)
         *   dx  = ∂f/∂x       (∂f/∂tx)
         *   dy  = ∂f/∂y       (∂f/∂ty)
         */
        float dx0, dx1, dy0, dy1, dx, dy;
        for (int i = 0; i < height; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx + ((float) i) * yvecx;
            coordy = currentoffsety + ((float) i) * yvecy;
            for (int n = 0; n < width; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < targetwidth) && (msky >= 0) && (msky < targetheight)) {
                    area++;

                    computeXInterpolationIndicesFloat(coordx, doubletargetwidth, targetwidth, xInterpolationIndices);
                    computeYInterpolationIndicesFloat(coordy, doubletargetheight, targetheight, targetwidth, yInterpolationIndices);

                    rescoordx = getFractionalFloat(coordx);
                    rescoordy = getFractionalFloat(coordy);
                    
                    s = interpolateCubicBSplineFloat(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, target);

                    /*
                     * Residual: r_i = f(x_i) - g(T_p(x_i))
                     * where f is the source image and g is the B-spline-interpolated target.
                     * We reuse rescoordx to store the residual (same as translation/rigid body).
                     */
                    float diff = source[nIndex] - s;
                    msqe += diff * diff;
                    
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
                    dx = xGradient[nIndex];
                    dy = yGradient[nIndex];
                    dx0 = ((float)n) * dx;   // n * ∂f/∂x
                    dx1 = ((float)i) * dx;   // i * ∂f/∂x
                    dy0 = ((float)n) * dy;   // n * ∂f/∂y
                    dy1 = ((float)i) * dy;   // i * ∂f/∂y
                    
                    /*
                     * TODO/FIXME/KNOWN ISSUE: The following summation is MUCH worse than the
                     * parallel sum reduction done on the GPU, because (relatively speaking) small
                     * numbers are added to an ever growing larger number reducing the precision in
                     * the outcome. Currently I ignore this like the original implementation, but
                     * this is one of many reasons why the GPU version and the CPU version will
                     * never yield the same results!
                     */

                    /*
                     * Gradient accumulation: g_k += r_i * J_ik
                     * Parameter order: (a11, a12, a21, a22, tx, ty)
                     */
                    fgradient[0] += diff * dx0;   // Σ r * n * ∂f/∂x
                    fgradient[1] += diff * dx1;   // Σ r * i * ∂f/∂x
                    fgradient[2] += diff * dy0;   // Σ r * n * ∂f/∂y
                    fgradient[3] += diff * dy1;   // Σ r * i * ∂f/∂y
                    fgradient[4] += diff * dx;    // Σ r * ∂f/∂x
                    fgradient[5] += diff * dy;    // Σ r * ∂f/∂y

                    /*
                     * Hessian accumulation (upper triangle only):
                     * H_kl += J_ik * J_il
                     *
                     * This is the Gauss-Newton approximation H ≈ J^T J,
                     * ignoring the second-order terms r_i * ∇²f_i.
                     * Only 21 unique entries (upper triangle of 6×6 symmetric matrix).
                     */
                    // Row 0: a11 × {a11, a12, a21, a22, tx, ty}
                    fhessian[0][0] += dx0 * dx0;
                    fhessian[0][1] += dx0 * dx1;
                    fhessian[0][2] += dx0 * dy0;
                    fhessian[0][3] += dx0 * dy1;
                    fhessian[0][4] += dx0 * dx;
                    fhessian[0][5] += dx0 * dy;
                    // Row 1: a12 × {a12, a21, a22, tx, ty}
                    fhessian[1][1] += dx1 * dx1;
                    fhessian[1][2] += dx1 * dy0;
                    fhessian[1][3] += dx1 * dy1;
                    fhessian[1][4] += dx1 * dx;
                    fhessian[1][5] += dx1 * dy;
                    // Row 2: a21 × {a21, a22, tx, ty}
                    fhessian[2][2] += dy0 * dy0;
                    fhessian[2][3] += dy0 * dy1;
                    fhessian[2][4] += dy0 * dx;
                    fhessian[2][5] += dy0 * dy;
                    // Row 3: a22 × {a22, tx, ty}
                    fhessian[3][3] += dy1 * dy1;
                    fhessian[3][4] += dy1 * dx;
                    fhessian[3][5] += dy1 * dy;
                    // Row 4: tx × {tx, ty}
                    fhessian[4][4] += dx * dx;
                    fhessian[4][5] += dx * dy;
                    // Row 5: ty × {ty}
                    fhessian[5][5] += dy * dy;
                }
                // walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
        gradient[0] = fgradient[0];
        gradient[1] = fgradient[1];
        gradient[2] = fgradient[2];
        gradient[3] = fgradient[3];
        gradient[4] = fgradient[4];
        gradient[5] = fgradient[5];
        hessian[0][0] = fhessian[0][0];
        hessian[0][1] = fhessian[0][1];
        hessian[0][2] = fhessian[0][2];
        hessian[0][3] = fhessian[0][3];
        hessian[0][4] = fhessian[0][4];
        hessian[0][5] = fhessian[0][5];
        hessian[1][1] = fhessian[1][1];
        hessian[1][2] = fhessian[1][2];
        hessian[1][3] = fhessian[1][3];
        hessian[1][4] = fhessian[1][4];
        hessian[1][5] = fhessian[1][5];
        hessian[2][2] = fhessian[2][2];
        hessian[2][3] = fhessian[2][3];
        hessian[2][4] = fhessian[2][4];
        hessian[2][5] = fhessian[2][5];
        hessian[3][3] = fhessian[3][3];
        hessian[3][4] = fhessian[3][4];
        hessian[3][5] = fhessian[3][5];
        hessian[4][4] = fhessian[4][4];
        hessian[4][5] = fhessian[4][5];
        hessian[5][5] = fhessian[5][5];
        // symmetrize hessian
        for (int i = 1; (i < 6); i++) {
            for (int j = 0; (j < i); j++) {
                hessian[i][j] = hessian[j][i];
            }
        }
        return msqe / ((double) area);
    }
    
    private double getTranslationMeanSquaresWithoutHessianDouble(double currentoffsetx, double currentoffsety) {
        final int width = (int) sharedContext.img.dimension(0);
        final int height = (int) sharedContext.img.dimension(1);
        final int doubletargetwidth = width * 2;
        final int doubletargetheight = height * 2;
        int nIndex = 0;
        int area = 0;
        double s;
        double msqe = 0.0;// mean square error
        double coordx;
        double rescoordx;
        double coordy;
        double rescoordy;
        int mskx;
        int msky;
        for (int i = 0; i < height; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx;
            coordy = currentoffsety + ((double) i);
            for (int n = 0; n < width; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < width) && (msky >= 0) && (msky < height)) {
                    area++;
                    
                    PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubletargetwidth, width, xInterpolationIndices);
                    PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubletargetheight, height, width, yInterpolationIndices);

                    rescoordx = PlainJavaCPUAligner.getFractional(coordx);
                    rescoordy = PlainJavaCPUAligner.getFractional(coordy);
                    
                    s = PlainJavaCPUAligner.interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, doubleTargetCoefficients);

                    // calculate the return values
                    double diff = doubleSourceImage[nIndex] - s;
                    msqe += diff * diff;
                }
                // walk along the X-vector direction
                coordx += 1.0;
            }
        }
        return msqe / ((double) area);
    }
    
    private double getRigidBodyMeanSquaresWithoutHessianDouble(double currentoffsetx, double currentoffsety,
            double currentangle) {
        final int width = (int) sharedContext.img.dimension(0);
        final int height = (int) sharedContext.img.dimension(1);
        final int doubletargetwidth = width * 2;
        final int doubletargetheight = height * 2;
        int nIndex = 0;
        int area = 0;
        double s;
        double msqe = 0.0;// mean square error
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
        for (int i = 0; i < height; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx + ((double) i) * yvecx;
            coordy = currentoffsety + ((double) i) * yvecy;
            for (int n = 0; n < width; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < width) && (msky >= 0) && (msky < height)) {
                    area++;
                    
                    PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubletargetwidth, width, xInterpolationIndices);
                    PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubletargetheight, height, width, yInterpolationIndices);

                    rescoordx = PlainJavaCPUAligner.getFractional(coordx);
                    rescoordy = PlainJavaCPUAligner.getFractional(coordy);
                    
                    s = PlainJavaCPUAligner.interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, doubleTargetCoefficients);

                    // calculate the return values
                    double diff = doubleSourceImage[nIndex] - s;
                    msqe += diff * diff;
                }
                // walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
        return msqe / ((double) area);
    }
    
    private double getScaledRotationMeanSquaresWithoutHessianDouble(double currentoffsetx, double currentoffsety,
            double currentangle, double currentscale) {
        final int width = (int) sharedContext.img.dimension(0);
        final int height = (int) sharedContext.img.dimension(1);
        final int doubletargetwidth = width * 2;
        final int doubletargetheight = height * 2;
        int nIndex = 0;
        int area = 0;
        double s;
        double msqe = 0.0;// mean square error
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
        for (int i = 0; i < height; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx + ((double) i) * yvecx;
            coordy = currentoffsety + ((double) i) * yvecy;
            for (int n = 0; n < width; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < width) && (msky >= 0) && (msky < height)) {
                    area++;
                    
                    PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubletargetwidth, width, xInterpolationIndices);
                    PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubletargetheight, height, width, yInterpolationIndices);

                    rescoordx = PlainJavaCPUAligner.getFractional(coordx);
                    rescoordy = PlainJavaCPUAligner.getFractional(coordy);
                    
                    s = PlainJavaCPUAligner.interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, doubleTargetCoefficients);

                    // calculate the return values
                    double diff = doubleSourceImage[nIndex] - s;
                    msqe += diff * diff;
                }
                // walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
        return msqe / ((double) area);
    }
    
    private double getAffineMeanSquaresWithoutHessianDouble(double currentoffsetx, double currentoffsety,
            double currenta11, double currenta12,
            double currenta21, double currenta22) {
        final int width = (int) sharedContext.img.dimension(0);
        final int height = (int) sharedContext.img.dimension(1);
        final int doubletargetwidth = width * 2;
        final int doubletargetheight = height * 2;
        int nIndex = 0;
        int area = 0;
        double s;
        double msqe = 0.0;// mean square error
        double xvecx = currenta11;
        double xvecy = currenta21;
        double yvecx = currenta12;
        double yvecy = currenta22;
        double coordx;
        double rescoordx;
        double coordy;
        double rescoordy;
        int mskx;
        int msky;
        for (int i = 0; i < height; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx + ((double) i) * yvecx;
            coordy = currentoffsety + ((double) i) * yvecy;
            for (int n = 0; n < width; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < width) && (msky >= 0) && (msky < height)) {
                    area++;
                    
                    PlainJavaCPUAligner.computeXInterpolationIndices(coordx, doubletargetwidth, width, xInterpolationIndices);
                    PlainJavaCPUAligner.computeYInterpolationIndices(coordy, doubletargetheight, height, width, yInterpolationIndices);

                    rescoordx = PlainJavaCPUAligner.getFractional(coordx);
                    rescoordy = PlainJavaCPUAligner.getFractional(coordy);
                    
                    s = PlainJavaCPUAligner.interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, doubleTargetCoefficients);

                    // calculate the return values
                    double diff = doubleSourceImage[nIndex] - s;
                    msqe += diff * diff;
                }
                // walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
        return msqe / ((double) area);
    }
    
    private double getTranslationMeanSquaresWithoutHessian(int pyramidIndex, float currentoffsetx, float currentoffsety) {
        final int width = (int) sourcePyramid[pyramidIndex].width;
        final int height = (int) sourcePyramid[pyramidIndex].height;
        final float[] source = sourcePyramid[pyramidIndex].Image;
        final int targetwidth = (int) targetPyramid[pyramidIndex].width;
        final int targetheight = (int) targetPyramid[pyramidIndex].height;
        final int doubletargetwidth = targetwidth * 2;
        final int doubletargetheight = targetheight * 2;
        final float[] target = targetPyramid[pyramidIndex].Coefficient;
        int nIndex = 0;
        int area = 0;
        float s;
        float msqe = 0.0f;// mean square error
        float coordx;
        float rescoordx;
        float coordy;
        float rescoordy;
        int mskx;
        int msky;
        for (int i = 0; i < height; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx;
            coordy = currentoffsety + ((float) i);
            for (int n = 0; n < width; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < targetwidth) && (msky >= 0) && (msky < targetheight)) {
                    area++;
                    
                    computeXInterpolationIndicesFloat(coordx, doubletargetwidth, targetwidth, xInterpolationIndices);
                    computeYInterpolationIndicesFloat(coordy, doubletargetheight, targetheight, targetwidth, yInterpolationIndices);

                    rescoordx = getFractionalFloat(coordx);
                    rescoordy = getFractionalFloat(coordy);
                    
                    s = interpolateCubicBSplineFloat(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, target);

                    // calculate the return values
                    float diff = source[nIndex] - s;
                    msqe += diff * diff;
                }
                // walk along the X-vector direction
                coordx += 1.0;
            }
        }
        return msqe / ((double) area);
    }

    private double getRigidBodyMeanSquaresWithoutHessian(int pyramidIndex, float currentoffsetx, float currentoffsety,
            float currentangle) {
        final int width = (int) sourcePyramid[pyramidIndex].width;
        final int height = (int) sourcePyramid[pyramidIndex].height;
        final float[] source = sourcePyramid[pyramidIndex].Image;
        final int targetwidth = (int) targetPyramid[pyramidIndex].width;
        final int targetheight = (int) targetPyramid[pyramidIndex].height;
        final int doubletargetwidth = targetwidth * 2;
        final int doubletargetheight = targetheight * 2;
        final float[] target = targetPyramid[pyramidIndex].Coefficient;
        int nIndex = 0;
        int area = 0;
        float s;
        float msqe = 0.0f;// mean square error
        float xvecx = (float) Math.cos(currentangle);
        float xvecy = (float) -Math.sin(currentangle);
        float yvecx = -xvecy;
        float yvecy = xvecx;
        float coordx;
        float rescoordx;
        float coordy;
        float rescoordy;
        int mskx;
        int msky;
        for (int i = 0; i < height; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx + ((float) i) * yvecx;
            coordy = currentoffsety + ((float) i) * yvecy;
            for (int n = 0; n < width; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < targetwidth) && (msky >= 0) && (msky < targetheight)) {
                    area++;
                    
                    computeXInterpolationIndicesFloat(coordx, doubletargetwidth, targetwidth, xInterpolationIndices);
                    computeYInterpolationIndicesFloat(coordy, doubletargetheight, targetheight, targetwidth, yInterpolationIndices);

                    rescoordx = getFractionalFloat(coordx);
                    rescoordy = getFractionalFloat(coordy);
                    
                    s = interpolateCubicBSplineFloat(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, target);

                    // calculate the return values
                    float diff = source[nIndex] - s;
                    msqe += diff * diff;
                }
                // walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
        return msqe / ((double) area);
    }
    
    private double getScaledRotationMeanSquaresWithoutHessian(int pyramidIndex, float currentoffsetx, float currentoffsety, float currentangle, float currentscale) {
        final int width = (int) sourcePyramid[pyramidIndex].width;
        final int height = (int) sourcePyramid[pyramidIndex].height;
        final float[] source = sourcePyramid[pyramidIndex].Image;
        final int targetwidth = (int) targetPyramid[pyramidIndex].width;
        final int targetheight = (int) targetPyramid[pyramidIndex].height;
        final int doubletargetwidth = targetwidth * 2;
        final int doubletargetheight = targetheight * 2;
        final float[] target = targetPyramid[pyramidIndex].Coefficient;
        int nIndex = 0;
        int area = 0;
        float s;
        float msqe = 0.0f;// mean square error
        float xvecx = (float)Math.cos(currentangle) * currentscale;
        float xvecy = (float)-Math.sin(currentangle) * currentscale;
        float yvecx = -xvecy;
        float yvecy = xvecx;
        float coordx;
        float rescoordx;
        float coordy;
        float rescoordy;
        int mskx;
        int msky;
        for (int i = 0; i < height; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx + ((float) i) * yvecx;
            coordy = currentoffsety + ((float) i) * yvecy;
            for (int n = 0; n < width; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < targetwidth) && (msky >= 0) && (msky < targetheight)) {
                    area++;
                    
                    computeXInterpolationIndicesFloat(coordx, doubletargetwidth, targetwidth, xInterpolationIndices);
                    computeYInterpolationIndicesFloat(coordy, doubletargetheight, targetheight, targetwidth, yInterpolationIndices);

                    rescoordx = getFractionalFloat(coordx);
                    rescoordy = getFractionalFloat(coordy);
                    
                    s = interpolateCubicBSplineFloat(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, target);

                    // calculate the return values
                    float diff = source[nIndex] - s;
                    msqe += diff * diff;
                }
                // walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
        return msqe / ((double) area);
    }
    
    private double getAffineMeanSquaresWithoutHessian(int pyramidIndex, float currentoffsetx, float currentoffsety,
    		float currenta11, float currenta12,
    		float currenta21, float currenta22) {
        final int width = (int) sourcePyramid[pyramidIndex].width;
        final int height = (int) sourcePyramid[pyramidIndex].height;
        final float[] source = sourcePyramid[pyramidIndex].Image;
        final int targetwidth = (int) targetPyramid[pyramidIndex].width;
        final int targetheight = (int) targetPyramid[pyramidIndex].height;
        final int doubletargetwidth = targetwidth * 2;
        final int doubletargetheight = targetheight * 2;
        final float[] target = targetPyramid[pyramidIndex].Coefficient;
        int nIndex = 0;
        int area = 0;
        float s;
        float msqe = 0.0f;// mean square error
        float xvecx = currenta11;
        float xvecy = currenta21;
        float yvecx = currenta12;
        float yvecy = currenta22;
        float coordx;
        float rescoordx;
        float coordy;
        float rescoordy;
        int mskx;
        int msky;
        for (int i = 0; i < height; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx + ((float) i) * yvecx;
            coordy = currentoffsety + ((float) i) * yvecy;
            for (int n = 0; n < width; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < targetwidth) && (msky >= 0) && (msky < targetheight)) {
                    area++;
                    
                    computeXInterpolationIndicesFloat(coordx, doubletargetwidth, targetwidth, xInterpolationIndices);
                    computeYInterpolationIndicesFloat(coordy, doubletargetheight, targetheight, targetwidth, yInterpolationIndices);

                    rescoordx = getFractionalFloat(coordx);
                    rescoordy = getFractionalFloat(coordy);
                    
                    s = interpolateCubicBSplineFloat(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, target);

                    // calculate the return values
                    float diff = source[nIndex] - s;
                    msqe += diff * diff;
                }
                // walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
        return msqe / ((double) area);
    }
}
