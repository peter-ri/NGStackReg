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
import ch.unibas.biozentrum.imagejplugins.util.RigidBodyTransformation;
import ch.unibas.biozentrum.imagejplugins.util.TranslationTransformation;
import ch.unibas.biozentrum.imagejplugins.util.StaticUtility;
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
    private final double[][] hessian;
    private final float[][] fhessian;
    private final double[][] pseudoHessian;
    private final double[] gradient;
    private final float[] fgradient;
    private final int[] xInterpolationIndices;
    private final int[] yInterpolationIndices;
    private final double[] xWeights;
    private final double[] yWeights;
    private final float[] fxWeights;
    private final float[] fyWeights;
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
        this.xWeights = new double[] { 0.0, 0.0, 0.0, 0.0 };
        this.yWeights = new double[] { 0.0, 0.0, 0.0, 0.0 };
        this.fxWeights = new float[] { 0.0f, 0.0f, 0.0f, 0.0f };
        this.fyWeights = new float[] { 0.0f, 0.0f, 0.0f, 0.0f };
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
        case SCALEDROTATION:
        case AFFINE:
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
        if (width * height > Integer.MAX_VALUE) {
            throw new RuntimeException("Cannot allocate more than " + Integer.MAX_VALUE);
        }
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
                break;
            case RIGIDBODY:
                ((RigidBodyTransformation) scat.transformation).angle = angle;
                ((RigidBodyTransformation) scat.transformation).offsetx = offsetx;
                ((RigidBodyTransformation) scat.transformation).offsety = offsety;
                break;
            case SCALEDROTATION:
                break;
            case AFFINE:
                break;
            }

            // Don't forget to reset the transformation values for the next image
            offsetx = 0.0;
            offsety = 0.0;
            angle = 0.0;
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
                    break;
                case RIGIDBODY:
                    ((RigidBodyTransformation) scat.transformation).angle = angle;
                    ((RigidBodyTransformation) scat.transformation).offsetx = offsetx;
                    ((RigidBodyTransformation) scat.transformation).offsety = offsety;
                    break;
                case SCALEDROTATION:
                    break;
                case AFFINE:
                    break;
                }

                // Don't forget to reset the transformation values for the next image
                offsetx = 0.0;
                offsety = 0.0;
                angle = 0.0;
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
                break;
            case AFFINE:
                break;
            }
            converter.deConvertTo(doublefullSizedHelperBuffer2, scat.targetArray);
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
        for (int i = 0; i < height; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx;
            coordy = currentoffsety + ((double) i);
            for (int n = 0; n < width; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < width) && (msky >= 0) && (msky < height)) {
                    // Calculate X-interpolation indices
                    p = (coordx >= 0) ? (((int) coordx) + 2) : (((int) coordx) + 1);
                    for (int c = 0; c < 4; c++, p--) {
                        q = (p < 0) ? (-1 - p) : p;
                        if (q >= doubleWidth) {
                            q -= (doubleWidth) * (q / (doubleWidth)); // Warning this is an integer division it doesn't
                                                                      // yield q
                        }
                        xInterpolationIndices[c] = q >= width ? (doubleWidth - 1 - q) : q;
                    }
                    // calculate Y-interpolation indices
                    p = (coordy >= 0) ? (((int) coordy) + 2) : (((int) coordy) + 1);
                    for (int c = 0; c < 4; c++, p--) {
                        q = (p < 0) ? (-1 - p) : p;
                        if (q >= doubleHeight) {
                            q -= (doubleHeight) * (q / (doubleHeight)); // Warning this is an integer division it
                                                                        // doesn't yield q
                        }
                        // calculate linearized coordinates of the coefficient array
                        yInterpolationIndices[c] = q >= height ? (doubleHeight - 1 - q) * width : q * width;
                    }
                    // get the residuals of the coordinates
                    rescoordx = coordx - (coordx >= 0.0 ? ((double) ((int) coordx)) : ((double) (((int) coordx) - 1)));
                    rescoordy = coordy - (coordy >= 0.0 ? ((double) ((int) coordy)) : ((double) (((int) coordy) - 1)));
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
                    for (int y = 0; y < 4; y++) {
                        rescoordx = 0.0;// To avoid using too many variables this one will be repurposed
                        tmpindex = yInterpolationIndices[y];
                        for (int x = 0; x < 4; x++) {
                            rescoordx += xWeights[x] * doublefullSizedHelperBuffer[tmpindex + xInterpolationIndices[x]];
                        }
                        s += yWeights[y] * rescoordx;
                    }
                    doublefullSizedHelperBuffer2[nIndex] = s;
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
        for (int i = 0; i < height; i++) {
            // First walk along the Y-vector direction and reset the X-position (otherwise the
            // y position is initially correct and then lagging behind by one all the time)
            coordx = currentoffsetx + ((double) i) * yvecx;
            coordy = currentoffsety + ((double) i) * yvecy;
            for (int n = 0; n < width; n++, nIndex++) {
                mskx = (int) Math.round(coordx);
                msky = (int) Math.round(coordy);
                if ((mskx >= 0) && (mskx < width) && (msky >= 0) && (msky < height)) {
                    // Calculate X-interpolation indices
                    p = (coordx >= 0) ? (((int) coordx) + 2) : (((int) coordx) + 1);
                    for (int c = 0; c < 4; c++, p--) {
                        q = (p < 0) ? (-1 - p) : p;
                        if (q >= doubleWidth) {
                            q -= (doubleWidth) * (q / (doubleWidth)); // Warning this is an integer division it doesn't
                                                                      // yield q
                        }
                        xInterpolationIndices[c] = q >= width ? (doubleWidth - 1 - q) : q;
                    }
                    // calculate Y-interpolation indices
                    p = (coordy >= 0) ? (((int) coordy) + 2) : (((int) coordy) + 1);
                    for (int c = 0; c < 4; c++, p--) {
                        q = (p < 0) ? (-1 - p) : p;
                        if (q >= doubleHeight) {
                            q -= (doubleHeight) * (q / (doubleHeight)); // Warning this is an integer division it
                                                                        // doesn't yield q
                        }
                        // calculate linearized coordinates of the coefficient array
                        yInterpolationIndices[c] = q >= height ? (doubleHeight - 1 - q) * width : q * width;
                    }
                    // get the residuals of the coordinates
                    rescoordx = coordx - (coordx >= 0.0 ? ((double) ((int) coordx)) : ((double) (((int) coordx) - 1)));
                    rescoordy = coordy - (coordy >= 0.0 ? ((double) ((int) coordy)) : ((double) (((int) coordy) - 1)));
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
                    for (int y = 0; y < 4; y++) {
                        rescoordx = 0.0;// To avoid using too many variables this one will be repurposed
                        tmpindex = yInterpolationIndices[y];
                        for (int x = 0; x < 4; x++) {
                            rescoordx += xWeights[x] * doublefullSizedHelperBuffer[tmpindex + xInterpolationIndices[x]];
                        }
                        s += yWeights[y] * rescoordx;
                    }
                    doublefullSizedHelperBuffer2[nIndex] = s;
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
            break;
        case AFFINE:
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

    private double getRigidBodyMeanSquaresDouble(double currentoffsetx, double currentoffsety, double currentangle) {
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
        final int width = (int) sharedContext.img.dimension(0);
        final int height = (int) sharedContext.img.dimension(1);
        final int doubletargetwidth = width * 2;
        final int doubletargetheight = height * 2;
        int nIndex = 0;
        int area = 0;
        int p;
        int q;
        int tmpindex;
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
                    // Calculate X-interpolation indices
                    p = (coordx >= 0) ? (((int) coordx) + 2) : (((int) coordx) + 1);
                    for (int c = 0; c < 4; c++, p--) {
                        q = (p < 0) ? (-1 - p) : p;
                        q = (q < doubletargetwidth) ? q : q % doubletargetwidth;
                        xInterpolationIndices[c] = q >= width ? (doubletargetwidth - 1 - q) : q;
                    }
                    // calculate Y-interpolation indices
                    p = (coordy >= 0) ? (((int) coordy) + 2) : (((int) coordy) + 1);
                    for (int c = 0; c < 4; c++, p--) {
                        q = (p < 0) ? (-1 - p) : p;
                        q = (q < doubletargetheight) ? q : q % doubletargetheight;
                        // calculate linearized coordinates of the coefficient array
                        yInterpolationIndices[c] = q >= height ? (doubletargetheight - 1 - q) * width : q * width;
                    }
                    // get the residuals of the coordinates
                    rescoordx = coordx - (coordx >= 0.0 ? ((double) ((int) coordx)) : ((double) (((int) coordx) - 1)));
                    rescoordy = coordy - (coordy >= 0.0 ? ((double) ((int) coordy)) : ((double) (((int) coordy) - 1)));
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
                    for (int y = 0; y < 4; y++) {
                        rescoordx = 0.0;// To avoid using too many variables this one will be repurposed
                        tmpindex = yInterpolationIndices[y];
                        for (int x = 0; x < 4; x++) {
                            rescoordx += xWeights[x] * doubleTargetCoefficients[tmpindex + xInterpolationIndices[x]];
                        }
                        s += yWeights[y] * rescoordx;
                    }

                    // calculate the values for returning
                    rescoordx = doubleSourceImage[nIndex] - s;// repurposed for diff
                    msqe += rescoordx * rescoordx;
                    // repurposed for Theta
                    rescoordy = doubleSourceyGradient[nIndex] * (double) n - doubleSourcexGradient[nIndex] * (double) i; 
                    /*
                     * TODO/FIXME/KNOWN ISSUE: The following summation is MUCH worse than the
                     * parallel sum reduction done on the GPU, because (relatively speaking) small
                     * numbers are added to an ever growing larger number reducing the precision in
                     * the outcome. Currently I ignore this like the original implementation, but
                     * this is one of many reasons why the GPU version and the CPU version will
                     * never yield the same results!
                     */
                    gradient[0] += rescoordx * rescoordy;
                    gradient[1] += rescoordx * doubleSourcexGradient[nIndex];
                    gradient[2] += rescoordx * doubleSourceyGradient[nIndex];
                    hessian[0][0] += rescoordy * rescoordy;
                    hessian[0][1] += rescoordy * doubleSourcexGradient[nIndex];
                    hessian[0][2] += rescoordy * doubleSourceyGradient[nIndex];
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
    
    
    private double getTranslationMeanSquaresDouble(double currentoffsetx, double currentoffsety) {
        // First reset the global values which will not be reset in the loop
        gradient[0] = 0.0;
        gradient[1] = 0.0;
        hessian[0][0] = 0.0;
        hessian[0][1] = 0.0;
        hessian[1][0] = 0.0;
        hessian[1][1] = 0.0;
        final int width = (int) sharedContext.img.dimension(0);
        final int height = (int) sharedContext.img.dimension(1);
        final int doubletargetwidth = width * 2;
        final int doubletargetheight = height * 2;
        int nIndex = 0;
        int area = 0;
        int p;
        int q;
        int tmpindex;
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
                    // Calculate X-interpolation indices
                    p = (coordx >= 0) ? (((int) coordx) + 2) : (((int) coordx) + 1);
                    for (int c = 0; c < 4; c++, p--) {
                        q = (p < 0) ? (-1 - p) : p;
                        q = (q < doubletargetwidth) ? q : q % doubletargetwidth;
                        xInterpolationIndices[c] = q >= width ? (doubletargetwidth - 1 - q) : q;
                    }
                    // calculate Y-interpolation indices
                    p = (coordy >= 0) ? (((int) coordy) + 2) : (((int) coordy) + 1);
                    for (int c = 0; c < 4; c++, p--) {
                        q = (p < 0) ? (-1 - p) : p;
                        q = (q < doubletargetheight) ? q : q % doubletargetheight;
                        // calculate linearized coordinates of the coefficient array
                        yInterpolationIndices[c] = q >= height ? (doubletargetheight - 1 - q) * width : q * width;
                    }
                    // get the residuals of the coordinates
                    rescoordx = coordx - (coordx >= 0.0 ? ((double) ((int) coordx)) : ((double) (((int) coordx) - 1)));
                    rescoordy = coordy - (coordy >= 0.0 ? ((double) ((int) coordy)) : ((double) (((int) coordy) - 1)));
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
                    for (int y = 0; y < 4; y++) {
                        rescoordx = 0.0;// To avoid using too many variables this one will be repurposed
                        tmpindex = yInterpolationIndices[y];
                        for (int x = 0; x < 4; x++) {
                            rescoordx += xWeights[x] * doubleTargetCoefficients[tmpindex + xInterpolationIndices[x]];
                        }
                        s += yWeights[y] * rescoordx;
                    }

                    // calculate the values for returning
                    rescoordx = doubleSourceImage[nIndex] - s;// repurposed for diff
                    msqe += rescoordx * rescoordx;
                    /*
                     * TODO/FIXME/KNOWN ISSUE: The following summation is MUCH worse than the
                     * parallel sum reduction done on the GPU, because (relatively speaking) small
                     * numbers are added to an ever growing larger number reducing the precision in
                     * the outcome. Currently I ignore this like the original implementation, but
                     * this is one of many reasons why the GPU version and the CPU version will
                     * never yield the same results!
                     */
                    gradient[0] += rescoordx * doubleSourcexGradient[nIndex];
                    gradient[1] += rescoordx * doubleSourceyGradient[nIndex];
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

    private double getRigidBodyMeanSquares(int pyramidIndex, float currentoffsetx, float currentoffsety,
            float currentangle) {
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
        fgradient[0] = 0.0f;
        fgradient[1] = 0.0f;
        fgradient[2] = 0.0f;
        fhessian[0][0] = 0.0f;
        fhessian[0][1] = 0.0f;
        fhessian[0][2] = 0.0f;
        fhessian[1][1] = 0.0f;
        fhessian[1][2] = 0.0f;
        fhessian[2][2] = 0.0f;
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
        int p;
        int q;
        int tmpindex;
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
                    // Calculate X-interpolation indices
                    p = (coordx >= 0) ? (((int) coordx) + 2) : (((int) coordx) + 1);
                    for (int c = 0; c < 4; c++, p--) {
                        q = (p < 0) ? (-1 - p) : p;
                        q = (q < doubletargetwidth) ? q : q % doubletargetwidth;
                        xInterpolationIndices[c] = q >= targetwidth ? (doubletargetwidth - 1 - q) : q;
                    }
                    // calculate Y-interpolation indices
                    p = (coordy >= 0) ? (((int) coordy) + 2) : (((int) coordy) + 1);
                    for (int c = 0; c < 4; c++, p--) {
                        q = (p < 0) ? (-1 - p) : p;
                        q = (q < doubletargetheight) ? q : q % doubletargetheight;
                        yInterpolationIndices[c] = q >= targetheight ? (doubletargetheight - 1 - q) * targetwidth
                                : q * targetwidth;// calculate linearized coordinates of the coefficient array
                    }
                    // get the residuals of the coordinates
                    rescoordx = coordx - (coordx >= 0.0 ? ((float) ((int) coordx)) : ((float) (((int) coordx) - 1)));
                    rescoordy = coordy - (coordy >= 0.0 ? ((float) ((int) coordy)) : ((float) (((int) coordy) - 1)));
                    // calculate the X-weights
                    s = 1.0f - rescoordx;
                    fxWeights[3] = (float) Math.pow(s, 3.0) / 6.0f;
                    s = rescoordx * rescoordx;
                    fxWeights[2] = (2.0f / 3.0f) - 0.5f * s * (2.0f - rescoordx);
                    fxWeights[0] = s * rescoordx / 6.0f;
                    fxWeights[1] = 1.0f - fxWeights[0] - fxWeights[2] - fxWeights[3];
                    // calculate the Y-weights
                    s = 1.0f - rescoordy;
                    fyWeights[3] = (float) Math.pow(s, 3.0) / 6.0f;
                    s = rescoordy * rescoordy;
                    fyWeights[2] = (2.0f / 3.0f) - 0.5f * s * (2.0f - rescoordy);
                    fyWeights[0] = s * rescoordy / 6.0f;
                    fyWeights[1] = 1.0f - fyWeights[0] - fyWeights[2] - fyWeights[3];

                    // calculate the interpolated value from the target coefficients
                    s = 0.0f;
                    for (int y = 0; y < 4; y++) {
                        rescoordx = 0.0f;// To avoid using too many variables this one will be repurposed
                        tmpindex = yInterpolationIndices[y];
                        for (int x = 0; x < 4; x++) {
                            rescoordx += fxWeights[x] * target[tmpindex + xInterpolationIndices[x]];
                        }
                        s += fyWeights[y] * rescoordx;
                    }

                    // calculate the values for returning
                    rescoordx = source[nIndex] - s;// repurposed for diff
                    msqe += rescoordx * rescoordx;
                    rescoordy = yGradient[nIndex] * (float) n - xGradient[nIndex] * (float) i;// repurposed for Theta
                    /*
                     * TODO/FIXME/KNOWN ISSUE: The following summation is MUCH worse than the
                     * parallel sum reduction done on the GPU, because (relatively speaking) small
                     * numbers are added to an ever growing larger number reducing the precision in
                     * the outcome. Currently I ignore this like the original implementation, but
                     * this is one of many reasons why the GPU version and the CPU version will
                     * never yield the same results!
                     */
                    fgradient[0] += rescoordx * rescoordy;
                    fgradient[1] += rescoordx * xGradient[nIndex];
                    fgradient[2] += rescoordx * yGradient[nIndex];
                    fhessian[0][0] += rescoordy * rescoordy;
                    fhessian[0][1] += rescoordy * xGradient[nIndex];
                    fhessian[0][2] += rescoordy * yGradient[nIndex];
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
        return msqe / ((float) area);
    }

    private double getTranslationMeanSquares(int pyramidIndex, float currentoffsetx, float currentoffsety) {
        // First reset the global values which will not be reset in the loop
        gradient[0] = 0.0;
        gradient[1] = 0.0;
        hessian[0][0] = 0.0;
        hessian[0][1] = 0.0;
        hessian[1][0] = 0.0;
        hessian[1][1] = 0.0;
        fgradient[0] = 0.0f;
        fgradient[1] = 0.0f;
        fhessian[0][0] = 0.0f;
        fhessian[0][1] = 0.0f;
        fhessian[1][0] = 0.0f;
        fhessian[1][1] = 0.0f;
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
        int p;
        int q;
        int tmpindex;
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
                    // Calculate X-interpolation indices
                    p = (coordx >= 0) ? (((int) coordx) + 2) : (((int) coordx) + 1);
                    for (int c = 0; c < 4; c++, p--) {
                        q = (p < 0) ? (-1 - p) : p;
                        q = (q < doubletargetwidth) ? q : q % doubletargetwidth;
                        xInterpolationIndices[c] = q >= targetwidth ? (doubletargetwidth - 1 - q) : q;
                    }
                    // calculate Y-interpolation indices
                    p = (coordy >= 0) ? (((int) coordy) + 2) : (((int) coordy) + 1);
                    for (int c = 0; c < 4; c++, p--) {
                        q = (p < 0) ? (-1 - p) : p;
                        q = (q < doubletargetheight) ? q : q % doubletargetheight;
                        yInterpolationIndices[c] = q >= targetheight ? (doubletargetheight - 1 - q) * targetwidth
                                : q * targetwidth;// calculate linearized coordinates of the coefficient array
                    }
                    // get the residuals of the coordinates
                    rescoordx = coordx - (coordx >= 0.0 ? ((float) ((int) coordx)) : ((float) (((int) coordx) - 1)));
                    rescoordy = coordy - (coordy >= 0.0 ? ((float) ((int) coordy)) : ((float) (((int) coordy) - 1)));
                    // calculate the X-weights
                    s = 1.0f - rescoordx;
                    fxWeights[3] = (float) Math.pow(s, 3.0) / 6.0f;
                    s = rescoordx * rescoordx;
                    fxWeights[2] = (2.0f / 3.0f) - 0.5f * s * (2.0f - rescoordx);
                    fxWeights[0] = s * rescoordx / 6.0f;
                    fxWeights[1] = 1.0f - fxWeights[0] - fxWeights[2] - fxWeights[3];
                    // calculate the Y-weights
                    s = 1.0f - rescoordy;
                    fyWeights[3] = (float) Math.pow(s, 3.0) / 6.0f;
                    s = rescoordy * rescoordy;
                    fyWeights[2] = (2.0f / 3.0f) - 0.5f * s * (2.0f - rescoordy);
                    fyWeights[0] = s * rescoordy / 6.0f;
                    fyWeights[1] = 1.0f - fyWeights[0] - fyWeights[2] - fyWeights[3];

                    // calculate the interpolated value from the target coefficients
                    s = 0.0f;
                    for (int y = 0; y < 4; y++) {
                        rescoordx = 0.0f;// To avoid using too many variables this one will be repurposed
                        tmpindex = yInterpolationIndices[y];
                        for (int x = 0; x < 4; x++) {
                            rescoordx += fxWeights[x] * target[tmpindex + xInterpolationIndices[x]];
                        }
                        s += fyWeights[y] * rescoordx;
                    }

                    // calculate the values for returning
                    rescoordx = source[nIndex] - s;// repurposed for diff
                    msqe += rescoordx * rescoordx;
                    /*
                     * TODO/FIXME/KNOWN ISSUE: The following summation is MUCH worse than the
                     * parallel sum reduction done on the GPU, because (relatively speaking) small
                     * numbers are added to an ever growing larger number reducing the precision in
                     * the outcome. Currently I ignore this like the original implementation, but
                     * this is one of many reasons why the GPU version and the CPU version will
                     * never yield the same results!
                     */
                    fgradient[0] += rescoordx * xGradient[nIndex];
                    fgradient[1] += rescoordx * yGradient[nIndex];
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
        return msqe / ((float) area);
    }
    
    private double getRigidBodyMeanSquaresWithoutHessianDouble(double currentoffsetx, double currentoffsety,
            double currentangle) {
        final int width = (int) sharedContext.img.dimension(0);
        final int height = (int) sharedContext.img.dimension(1);
        final int doubletargetwidth = width * 2;
        final int doubletargetheight = height * 2;
        int nIndex = 0;
        int area = 0;
        int p;
        int q;
        int tmpindex;
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
                    // Calculate X-interpolation indices
                    p = (coordx >= 0) ? (((int) coordx) + 2) : (((int) coordx) + 1);
                    for (int c = 0; c < 4; c++, p--) {
                        q = (p < 0) ? (-1 - p) : p;
                        q = q < doubletargetwidth ? q : q % doubletargetwidth;
                        xInterpolationIndices[c] = q >= width ? (doubletargetwidth - 1 - q) : q;
                    }
                    // calculate Y-interpolation indices
                    p = (coordy >= 0) ? (((int) coordy) + 2) : (((int) coordy) + 1);
                    for (int c = 0; c < 4; c++, p--) {
                        q = (p < 0) ? (-1 - p) : p;
                        q = q < doubletargetheight ? q : q % doubletargetheight;
                        // calculate linearized coordinates of the coefficient array
                        yInterpolationIndices[c] = q >= height ? (doubletargetheight - 1 - q) * width : q * width;
                    }
                    // get the residuals of the coordinates
                    rescoordx = coordx - (coordx >= 0.0 ? ((double) ((int) coordx)) : ((double) (((int) coordx) - 1)));
                    rescoordy = coordy - (coordy >= 0.0 ? ((double) ((int) coordy)) : ((double) (((int) coordy) - 1)));
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
                    for (int y = 0; y < 4; y++) {
                        rescoordx = 0.0;// To avoid using too many variables this one will be repurposed
                        tmpindex = yInterpolationIndices[y];
                        for (int x = 0; x < 4; x++) {
                            rescoordx += xWeights[x] * doubleTargetCoefficients[tmpindex + xInterpolationIndices[x]];
                        }
                        s += yWeights[y] * rescoordx;
                    }

                    // calculate the values for returning
                    rescoordx = doubleSourceImage[nIndex] - s;// repurposed for diff
                    msqe += rescoordx * rescoordx;
                }
                // walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
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
        int p;
        int q;
        int tmpindex;
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
                    // Calculate X-interpolation indices
                    p = (coordx >= 0) ? (((int) coordx) + 2) : (((int) coordx) + 1);
                    for (int c = 0; c < 4; c++, p--) {
                        q = (p < 0) ? (-1 - p) : p;
                        q = q < doubletargetwidth ? q : q % doubletargetwidth;
                        xInterpolationIndices[c] = q >= width ? (doubletargetwidth - 1 - q) : q;
                    }
                    // calculate Y-interpolation indices
                    p = (coordy >= 0) ? (((int) coordy) + 2) : (((int) coordy) + 1);
                    for (int c = 0; c < 4; c++, p--) {
                        q = (p < 0) ? (-1 - p) : p;
                        q = q < doubletargetheight ? q : q % doubletargetheight;
                        // calculate linearized coordinates of the coefficient array
                        yInterpolationIndices[c] = q >= height ? (doubletargetheight - 1 - q) * width : q * width;
                    }
                    // get the residuals of the coordinates
                    rescoordx = coordx - (coordx >= 0.0 ? ((double) ((int) coordx)) : ((double) (((int) coordx) - 1)));
                    rescoordy = coordy - (coordy >= 0.0 ? ((double) ((int) coordy)) : ((double) (((int) coordy) - 1)));
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
                    for (int y = 0; y < 4; y++) {
                        rescoordx = 0.0;// To avoid using too many variables this one will be repurposed
                        tmpindex = yInterpolationIndices[y];
                        for (int x = 0; x < 4; x++) {
                            rescoordx += xWeights[x] * doubleTargetCoefficients[tmpindex + xInterpolationIndices[x]];
                        }
                        s += yWeights[y] * rescoordx;
                    }

                    // calculate the values for returning
                    rescoordx = doubleSourceImage[nIndex] - s;// repurposed for diff
                    msqe += rescoordx * rescoordx;
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
        int p;
        int q;
        int tmpindex;
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
                    // Calculate X-interpolation indices
                    p = (coordx >= 0) ? (((int) coordx) + 2) : (((int) coordx) + 1);
                    for (int c = 0; c < 4; c++, p--) {
                        q = (p < 0) ? (-1 - p) : p;
                        q = q < doubletargetwidth ? q : q % doubletargetwidth;
                        xInterpolationIndices[c] = q >= targetwidth ? (doubletargetwidth - 1 - q) : q;
                    }
                    // calculate Y-interpolation indices
                    p = (coordy >= 0) ? (((int) coordy) + 2) : (((int) coordy) + 1);
                    for (int c = 0; c < 4; c++, p--) {
                        q = (p < 0) ? (-1 - p) : p;
                        q = q < doubletargetheight ? q : q % doubletargetheight;
                        yInterpolationIndices[c] = q >= targetheight ? (doubletargetheight - 1 - q) * targetwidth
                                : q * targetwidth;// calculate linearized coordinates of the coefficient array
                    }
                    // get the residuals of the coordinates
                    rescoordx = coordx - (coordx >= 0.0f ? ((float) ((int) coordx)) : ((float) (((int) coordx) - 1)));
                    rescoordy = coordy - (coordy >= 0.0f ? ((float) ((int) coordy)) : ((float) (((int) coordy) - 1)));
                    // calculate the X-weights
                    s = 1.0f - rescoordx;
                    fxWeights[3] = (float) Math.pow(s, 3.0) / 6.0f;
                    s = rescoordx * rescoordx;
                    fxWeights[2] = (2.0f / 3.0f) - 0.5f * s * (2.0f - rescoordx);
                    fxWeights[0] = s * rescoordx / 6.0f;
                    fxWeights[1] = 1.0f - fxWeights[0] - fxWeights[2] - fxWeights[3];
                    // calculate the Y-weights
                    s = 1.0f - rescoordy;
                    fyWeights[3] = (float) Math.pow(s, 3.0) / 6.0f;
                    s = rescoordy * rescoordy;
                    fyWeights[2] = (2.0f / 3.0f) - 0.5f * s * (2.0f - rescoordy);
                    fyWeights[0] = s * rescoordy / 6.0f;
                    fyWeights[1] = 1.0f - fyWeights[0] - fyWeights[2] - fyWeights[3];

                    // calculate the interpolated value from the target coefficients
                    s = 0.0f;
                    for (int y = 0; y < 4; y++) {
                        rescoordx = 0.0f;// To avoid using too many variables this one will be repurposed
                        tmpindex = yInterpolationIndices[y];
                        for (int x = 0; x < 4; x++) {
                            rescoordx += fxWeights[x] * target[tmpindex + xInterpolationIndices[x]];
                        }
                        s += fyWeights[y] * rescoordx;
                    }

                    // calculate the values for returning
                    rescoordx = source[nIndex] - s;// repurposed for diff
                    msqe += rescoordx * rescoordx;
                }
                // walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
        return msqe / ((float) area);
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
        int p;
        int q;
        int tmpindex;
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
                    // Calculate X-interpolation indices
                    p = (coordx >= 0) ? (((int) coordx) + 2) : (((int) coordx) + 1);
                    for (int c = 0; c < 4; c++, p--) {
                        q = (p < 0) ? (-1 - p) : p;
                        q = q < doubletargetwidth ? q : q % doubletargetwidth;
                        xInterpolationIndices[c] = q >= targetwidth ? (doubletargetwidth - 1 - q) : q;
                    }
                    // calculate Y-interpolation indices
                    p = (coordy >= 0) ? (((int) coordy) + 2) : (((int) coordy) + 1);
                    for (int c = 0; c < 4; c++, p--) {
                        q = (p < 0) ? (-1 - p) : p;
                        q = q < doubletargetheight ? q : q % doubletargetheight;
                        yInterpolationIndices[c] = q >= targetheight ? (doubletargetheight - 1 - q) * targetwidth
                                : q * targetwidth;// calculate linearized coordinates of the coefficient array
                    }
                    // get the residuals of the coordinates
                    rescoordx = coordx - (coordx >= 0.0f ? ((float) ((int) coordx)) : ((float) (((int) coordx) - 1)));
                    rescoordy = coordy - (coordy >= 0.0f ? ((float) ((int) coordy)) : ((float) (((int) coordy) - 1)));
                    // calculate the X-weights
                    s = 1.0f - rescoordx;
                    fxWeights[3] = (float) Math.pow(s, 3.0) / 6.0f;
                    s = rescoordx * rescoordx;
                    fxWeights[2] = (2.0f / 3.0f) - 0.5f * s * (2.0f - rescoordx);
                    fxWeights[0] = s * rescoordx / 6.0f;
                    fxWeights[1] = 1.0f - fxWeights[0] - fxWeights[2] - fxWeights[3];
                    // calculate the Y-weights
                    s = 1.0f - rescoordy;
                    fyWeights[3] = (float) Math.pow(s, 3.0) / 6.0f;
                    s = rescoordy * rescoordy;
                    fyWeights[2] = (2.0f / 3.0f) - 0.5f * s * (2.0f - rescoordy);
                    fyWeights[0] = s * rescoordy / 6.0f;
                    fyWeights[1] = 1.0f - fyWeights[0] - fyWeights[2] - fyWeights[3];

                    // calculate the interpolated value from the target coefficients
                    s = 0.0f;
                    for (int y = 0; y < 4; y++) {
                        rescoordx = 0.0f;// To avoid using too many variables this one will be repurposed
                        tmpindex = yInterpolationIndices[y];
                        for (int x = 0; x < 4; x++) {
                            rescoordx += fxWeights[x] * target[tmpindex + xInterpolationIndices[x]];
                        }
                        s += fyWeights[y] * rescoordx;
                    }

                    // calculate the values for returning
                    rescoordx = source[nIndex] - s;// repurposed for diff
                    msqe += rescoordx * rescoordx;
                }
                // walk along the X-vector direction
                coordx += 1.0;
            }
        }
        return msqe / ((float) area);
    }
}
