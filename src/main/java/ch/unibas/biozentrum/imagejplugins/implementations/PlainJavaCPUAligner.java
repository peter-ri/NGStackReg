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
class MTNGSourcePyramidSlice
{
    long width;
    long height;
    double[] Image;
    double[] xGradient;
    double[] yGradient;
}
class MTNGTargetPyramidSlice
{
    long width;
    long height;
    double[] Coefficient;
}

public class PlainJavaCPUAligner extends CPUAligner
{
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
    private final double[][] pseudoHessian;
    private final double[] gradient;
    private final int[] xInterpolationIndices;
    private final int[] yInterpolationIndices;
    private int iterationPower;

    
    private MTNGSourcePyramidSlice[] sourcePyramid;
    private MTNGTargetPyramidSlice[] targetPyramid;
    
    private double[] entryImageBuffers;
    private double[] fullSizedHelperBuffer;
    
    private final SharedContextAlignmentTarget scat = new SharedContextAlignmentTarget();
    
    PlainJavaCPUAligner(final AbstractSharedContext sharedContext, final ImageConverter converter, final int pyramidDepth)
    {
        super(sharedContext, converter, pyramidDepth);
        this.xInterpolationIndices = new int[]{ 0,0,0,0 };
        this.yInterpolationIndices = new int[]{ 0,0,0,0 };
        switch(sharedContext.transformationType) {
            case TRANSLATION:
                this.gradient = new double[]{0.0, 0.0};
                this.hessian = new double[][]{{0.0,0.0}, {0.0,0.0}};
                this.pseudoHessian = new double[][]{{0.0,0.0}, {0.0,0.0}};
                break;
            case RIGIDBODY:
                this.gradient = new double[]{0.0, 0.0, 0.0};
                this.hessian = new double[][]{{0.0,0.0,0.0}, {0.0,0.0,0.0}, {0.0,0.0,0.0}};
                this.pseudoHessian = new double[][]{{0.0,0.0,0.0}, {0.0,0.0,0.0}, {0.0,0.0,0.0}};
                break;
            case SCALEDROTATION:
                this.gradient = new double[]{0.0, 0.0, 0.0, 0.0};
                this.hessian = new double[][]{{0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0}};
                this.pseudoHessian = new double[][]{{0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0}};
                break;
            case AFFINE:
                this.gradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                this.hessian = new double[][]{{0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}};
                this.pseudoHessian = new double[][]{{0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,0.0}};
                break;
            default:
                this.gradient = new double[]{0.0, 0.0, 0.0};
                this.hessian = new double[][]{{0.0,0.0,0.0}, {0.0,0.0,0.0}, {0.0,0.0,0.0}};
                this.pseudoHessian = new double[][]{{0.0,0.0,0.0}, {0.0,0.0,0.0}, {0.0,0.0,0.0}};
                break;
        }
        allocateMemory();
    }

    private void allocateMemory()
    {
        long width = sharedContext.img.dimension(0);
        long height = sharedContext.img.dimension(1);
        if(width*height > Integer.MAX_VALUE)
        {
            throw new RuntimeException("Cannot allocate more than " + Integer.MAX_VALUE);
        }
        sourcePyramid = new MTNGSourcePyramidSlice[pyramidDepth];
        targetPyramid = new MTNGTargetPyramidSlice[pyramidDepth];
        entryImageBuffers = new double[(int)(width*height)];
        fullSizedHelperBuffer = new double[(int)(width*height)];

        for(int j = 0;j < pyramidDepth; j++)
        {
            sourcePyramid[j] = new MTNGSourcePyramidSlice();
            targetPyramid[j] = new MTNGTargetPyramidSlice();
            sourcePyramid[j].width = width;
            sourcePyramid[j].height = height;
            sourcePyramid[j].Image = new double[(int)(width*height)];
            sourcePyramid[j].xGradient = new double[(int)(width*height)];
            sourcePyramid[j].yGradient = new double[(int)(width*height)];
            targetPyramid[j].width = width;
            targetPyramid[j].height = height;
            targetPyramid[j].Coefficient = new double[(int)(width*height)];
            width /= 2;
            height /= 2;
        }
    }
    
    private void resizeAllocatedBuffers()
    {
        /*
         * This means the target buffer has to be resized and we can get rid of the remaining buffers
         * scat.targetArray -> entryImageBuffers (can stay the same)
         * entryImageBuffers -> fullSizedHelperBuffer (has to be resized)
         * fullSizedHelperBuffer -> scat.targetArray
         */
        long width = sharedContext.resizedTargetImage.dimension(0);
        long height = sharedContext.resizedTargetImage.dimension(1);
        
        if(width*height > Integer.MAX_VALUE)
        {
            throw new RuntimeException("Cannot allocate more than " + Integer.MAX_VALUE);
        }
        
        for(int j = 0;j < pyramidDepth; j++)
        {
            sourcePyramid[j].Image = null;
            sourcePyramid[j].xGradient = null;
            sourcePyramid[j].yGradient = null;
            targetPyramid[j].Coefficient = null;
        }
        sourcePyramid = null;
        targetPyramid = null;
        
        fullSizedHelperBuffer = new double[(int)(width*height)];
    }

    @Override
    public void run()
    {
        while(!sharedContext.getNextAlignmentTarget(scat))
        {
            // Construct the target pyramid
            constructTargetImagePyramid();
            // Construct the source image and derivative pyramids
            constructSourceImagePyramid();
            // Now that the pyramid has been constructed the optimization can commence

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
                // Construct the target pyramid
                constructTargetImagePyramid();
                // Construct the source image and derivative pyramids
                constructSourceImagePyramid();
                // Now that the pyramid has been constructed the optimization can commence

                doRegistration();
                switch(sharedContext.transformationType)
                {
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
                sharedContext.workerSynchronizationBarrier.await(); // This also combines all the transformations and resets the position (see the action implementation)
            } catch (InterruptedException ex) {
                throw new RuntimeException("Thread interrupted.");
            } catch (BrokenBarrierException ex) {
                throw new RuntimeException("Worker synchronization barrier is broken.");
            }
        }        
        
        if(sharedContext.getResizeAfterRegistration())
        {
            resizeAllocatedBuffers();
            // The following code requires the current position to have been reset to 0vector
            while(!sharedContext.getTransformationForCurrentPosition(scat)) {
                int width = (int)sharedContext.img.dimension(0);
                int height = (int)sharedContext.img.dimension(1);
                converter.convertTo(scat.targetArray, entryImageBuffers);
                // pre-multiply the image for cubic spline interpolation
                premultiplyCubicBSpline(entryImageBuffers, width * height);
                // Conversion to B-spline coefficients along X axis
                cubicBSplinePrefilter2DXhp(entryImageBuffers, width, height);
                // pre-multiply again
                premultiplyCubicBSpline(entryImageBuffers, width * height);
                // Now along the Y-axis
                cubicBSplinePrefilter2DYhp(entryImageBuffers, width, height);
                
                //TODO: These functions could be abstract methods which are implemented with the correct transformation saving a switch every single loop iteration also streamlining the code
                switch (sharedContext.transformationType) {
                    case TRANSLATION:
                        resizeTransformTranslateImageWithBsplineInterpolation(width, height, (int)sharedContext.resizedTargetImage.dimension(0), (int)sharedContext.resizedTargetImage.dimension(1), ((TranslationTransformation)scat.transformation).offsetx, ((TranslationTransformation)scat.transformation).offsety);
                        break;
                    case RIGIDBODY:
                        resizeTransformImageWithBsplineInterpolation(width, height, (int)sharedContext.resizedTargetImage.dimension(0), (int)sharedContext.resizedTargetImage.dimension(1), ((RigidBodyTransformation)scat.transformation).offsetx, ((RigidBodyTransformation)scat.transformation).offsety, ((RigidBodyTransformation)scat.transformation).angle);
                        break;
                    case SCALEDROTATION:
                        resizeScaledRotationImageWithBsplineInterpolation(width, height, (int)sharedContext.resizedTargetImage.dimension(0), (int)sharedContext.resizedTargetImage.dimension(1), ((ScaledRotationTransformation)scat.transformation).offsetx, ((ScaledRotationTransformation)scat.transformation).offsety, ((ScaledRotationTransformation)scat.transformation).angle, ((ScaledRotationTransformation)scat.transformation).scale);
                        break;
                    case AFFINE:
                        resizeAffineImageWithBsplineInterpolation(width, height, (int)sharedContext.resizedTargetImage.dimension(0), (int)sharedContext.resizedTargetImage.dimension(1), ((AffineTransformation)scat.transformation).offsetx, ((AffineTransformation)scat.transformation).offsety, ((AffineTransformation)scat.transformation).a11, ((AffineTransformation)scat.transformation).a12, ((AffineTransformation)scat.transformation).a21, ((AffineTransformation)scat.transformation).a22);
                        break;
                }
                
                converter.deConvertTo(fullSizedHelperBuffer,scat.sourceArray);
            }
        }
        else
        {
            // The following code requires the current position to have been reset to 0vector
            while(!sharedContext.getTransformationForCurrentPosition(scat)) {
                int width = (int)sharedContext.img.dimension(0);
                int height = (int)sharedContext.img.dimension(1);
                converter.convertTo(scat.targetArray, entryImageBuffers);
                // pre-multiply the image for cubic spline interpolation
                premultiplyCubicBSpline(entryImageBuffers, width * height);
                // Conversion to B-spline coefficients along X axis
                cubicBSplinePrefilter2DXhp(entryImageBuffers, width, height);
                // pre-multiply again
                premultiplyCubicBSpline(entryImageBuffers, width * height);
                // Now along the Y-axis
                cubicBSplinePrefilter2DYhp(entryImageBuffers, width, height);
                
                //TODO: These functions could be abstract methods which are implemented with the correct transformation saving a switch every single loop iteration also streamlining the code
                switch (sharedContext.transformationType) {
                    case TRANSLATION:
                        transformTranslateImageWithBsplineInterpolation(width,height, ((TranslationTransformation)scat.transformation).offsetx, ((TranslationTransformation)scat.transformation).offsety);
                        break;
                    case RIGIDBODY:
                        transformImageWithBsplineInterpolation(width,height, ((RigidBodyTransformation)scat.transformation).offsetx, ((RigidBodyTransformation)scat.transformation).offsety, ((RigidBodyTransformation)scat.transformation).angle);
                        break;
                    case SCALEDROTATION:
                        transformScaledRotationWithBsplineInterpolation(width,height, ((ScaledRotationTransformation)scat.transformation).offsetx, ((ScaledRotationTransformation)scat.transformation).offsety, ((ScaledRotationTransformation)scat.transformation).angle, ((ScaledRotationTransformation)scat.transformation).scale);
                        break;
                    case AFFINE:
                        transformAffineWithBsplineInterpolation(width,height, ((AffineTransformation)scat.transformation).offsetx, ((AffineTransformation)scat.transformation).offsety, ((AffineTransformation)scat.transformation).a11, ((AffineTransformation)scat.transformation).a12, ((AffineTransformation)scat.transformation).a21, ((AffineTransformation)scat.transformation).a22);
                        break;
                }
                
                converter.deConvertTo(fullSizedHelperBuffer,scat.targetArray);
            }
        }
    }

    private void transformTranslateImageWithBsplineInterpolation(final int width, final int height, double currentoffsetx, double currentoffsety)
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
                    computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                    computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                    
                    fullSizedHelperBuffer[nIndex] = interpolateCubicBSpline(getFractional(coordx), getFractional(coordy), xInterpolationIndices, yInterpolationIndices, entryImageBuffers);
                }
                else
                {
                    fullSizedHelperBuffer[nIndex] = 0.0;
                }
                // walk along the X-vector direction
                coordx += 1.0;
            }
        }
    }
    
    private void resizeTransformTranslateImageWithBsplineInterpolation(final int width, final int height, final int targetwidth, final int targetheight, double currentoffsetx, double currentoffsety)
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
                    computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                    computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                    
                    fullSizedHelperBuffer[nIndex] = interpolateCubicBSpline(getFractional(coordx), getFractional(coordy), xInterpolationIndices, yInterpolationIndices, entryImageBuffers);
                }
                else
                {
                    fullSizedHelperBuffer[nIndex] = 0.0;
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
                    computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                    computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                    
                    fullSizedHelperBuffer[nIndex] = interpolateCubicBSpline(getFractional(coordx), getFractional(coordy), xInterpolationIndices, yInterpolationIndices, entryImageBuffers);
                }
                else
                {
                    fullSizedHelperBuffer[nIndex] = 0.0;
                }
                // walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
    }
    
    private void resizeTransformImageWithBsplineInterpolation(final int width, final int height, final int targetwidth, final int targetheight, double currentoffsetx, double currentoffsety, double currentangle)
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
                    computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                    computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                    
                    fullSizedHelperBuffer[nIndex] = interpolateCubicBSpline(getFractional(coordx), getFractional(coordy), xInterpolationIndices, yInterpolationIndices, entryImageBuffers);
                }
                else
                {
                    fullSizedHelperBuffer[nIndex] = 0.0;
                }
                // walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
    }
    
    private void transformScaledRotationWithBsplineInterpolation(final int width, final int height, double currentoffsetx, double currentoffsety, double currentangle, double currentscale)
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
                    computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                    computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                    
                    fullSizedHelperBuffer[nIndex] = interpolateCubicBSpline(getFractional(coordx), getFractional(coordy), xInterpolationIndices, yInterpolationIndices, entryImageBuffers);
                }
                else
                {
                    fullSizedHelperBuffer[nIndex] = 0.0;
                }
                // walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
    }
    
    private void resizeScaledRotationImageWithBsplineInterpolation(final int width, final int height, final int targetwidth, final int targetheight, double currentoffsetx, double currentoffsety, double currentangle, double currentscale)
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
                    computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                    computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                    
                    fullSizedHelperBuffer[nIndex] = interpolateCubicBSpline(getFractional(coordx), getFractional(coordy), xInterpolationIndices, yInterpolationIndices, entryImageBuffers);
                }
                else
                {
                    fullSizedHelperBuffer[nIndex] = 0.0;
                }
                // walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
    }
    
    private void transformAffineWithBsplineInterpolation(final int width, final int height, double currentoffsetx, double currentoffsety, double currenta11, double currenta12, double currenta21, double currenta22)
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
                    computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                    computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                    
                    fullSizedHelperBuffer[nIndex] = interpolateCubicBSpline(getFractional(coordx), getFractional(coordy), xInterpolationIndices, yInterpolationIndices, entryImageBuffers);
                }
                else
                {
                    fullSizedHelperBuffer[nIndex] = 0.0;
                }
                // walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
    }
    
    private void resizeAffineImageWithBsplineInterpolation(final int width, final int height, final int targetwidth, final int targetheight, double currentoffsetx, double currentoffsety, double currenta11, double currenta12, double currenta21, double currenta22)
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
                    computeXInterpolationIndices(coordx, doubleWidth, width, xInterpolationIndices);
                    computeYInterpolationIndices(coordy, doubleHeight, height, width, yInterpolationIndices);
                    
                    fullSizedHelperBuffer[nIndex] = interpolateCubicBSpline(getFractional(coordx), getFractional(coordy), xInterpolationIndices, yInterpolationIndices, entryImageBuffers);
                }
                else
                {
                    fullSizedHelperBuffer[nIndex] = 0.0;
                }
                // walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
    }
    
    public static void premultiplyCubicBSpline(final double target[], final int nrOfElements)
    {
        for(int i = 0;i < nrOfElements;i++)
        {
            target[i] *= 6.0;
        }
    }
    public static void premultiplyCubicBSplineDeg7(final double target[], final int nrOfElements)
    {
        for(int i = 0;i < nrOfElements;i++)
        {
            target[i] *= MTNGStackReg.lambda7;
        }
    }
    public static void cubicBSplinePrefilter2DXhp(final double target[], final int width, final int height)
    {
        for(int i = 0;i < height;i++)
        {
            // causal initialization
            double z1 = MTNGStackReg.pole;
            double zn = Math.pow(z1, (double)width);
            double sum = (1.0 + MTNGStackReg.pole) * (target[i*width]+ zn * target[i*width + (width - 1)]);
            zn*=zn;
            for(int n = 1;n < width - 1;n++)
            {
                z1 *= MTNGStackReg.pole;
                zn /= MTNGStackReg.pole;
                sum += (z1 + zn) * target[i*width + n];
            }
            target[i*width] = (sum / (1.0 - Math.pow(MTNGStackReg.pole, (double)(2*width))));
            // causal recursion
            for(int n = 1;n < width;n++)
            {
                target[i*width+n] += MTNGStackReg.pole * target[i*width+n-1];
            }
            // anticausal initialization
            target[i*width + (width - 1)] = (MTNGStackReg.pole * target[i*width + (width - 1)] / (MTNGStackReg.pole - 1.0));

            // anticausal recursion
            for(int n = width - 2;n >= 0;n--)
            {
                target[i*width + n] = MTNGStackReg.pole * (target[i*width + n + 1] - target[i*width + n]);
            }
        }
    }
    public static void cubicBSplinePrefilter2DXhpDeg7(final double target[], final int width, final int height)
    {
        for(int p = 0;p < 3;p++)
        {
            for(int i = 0;i < height;i++)
            {
                // causal initialization
                double z1 = MTNGStackReg.polesDeg7[p];
                double zn = Math.pow(z1, (double)width);
                double sum = (1.0 + MTNGStackReg.polesDeg7[p]) * (target[i*width]+ zn * target[i*width + (width - 1)]);
                zn*=zn;
                for(int n = 1;n < width - 1;n++)
                {
                    z1 *= MTNGStackReg.polesDeg7[p];
                    zn /= MTNGStackReg.polesDeg7[p];
                    sum += (z1 + zn) * target[i*width + n];
                }
                target[i*width] = (sum / (1.0 - Math.pow(MTNGStackReg.polesDeg7[p], (double)(2*width))));
                // causal recursion
                for(int n = 1;n < width;n++)
                {
                    target[i*width+n] += MTNGStackReg.polesDeg7[p] * target[i*width+n-1];
                }
                // anticausal initialization
                target[i*width + (width - 1)] = (MTNGStackReg.polesDeg7[p] * target[i*width + (width - 1)] / (MTNGStackReg.polesDeg7[p] - 1.0));

                // anticausal recursion
                for(int n = width - 2;n >= 0;n--)
                {
                    target[i*width + n] = MTNGStackReg.polesDeg7[p] * (target[i*width + n + 1] - target[i*width + n]);
                }
            }
        }
    }
    public static void cubicBSplinePrefilter2DYhp(final double target[], final int width, final int height)
    {
        for(int i = 0;i < width;i++)
        {
            // causal initialization
            double z1 = MTNGStackReg.pole;
            double zn = Math.pow(z1, (double)height);
            double sum = (1.0 + MTNGStackReg.pole) * (target[i]+ zn * target[(height-1)*width + i]);
            zn*=zn;
            for(int n = 1;n < height - 1;n++)
            {
                z1 *= MTNGStackReg.pole;
                zn /= MTNGStackReg.pole;
                sum += (z1 + zn) * target[n*width + i];
            }
            target[i] = (sum / (1.0 - Math.pow(MTNGStackReg.pole, (double)(2*height))));
            // causal recursion
            for(int n = 1;n < height;n++)
            {
                target[n*width+i] += MTNGStackReg.pole * target[(n-1)*width+i];
            }
            // anticausal initialization
            target[(height-1)*width + i] = (MTNGStackReg.pole * target[(height-1)*width + i] / (MTNGStackReg.pole - 1.0));

            // anticausal recursion
            for(int n = height - 2;n >= 0;n--)
            {
                target[n*width + i] = MTNGStackReg.pole * (target[(n+1)*width + i] - target[n*width + i]);
            }
        }
    }
    public static void cubicBSplinePrefilter2DYhpDeg7(final double target[], final int width, final int height)
    {
        for(int p = 0;p < 3;p++)
        {
            for(int i = 0;i < width;i++)
            {
                // causal initialization
                double z1 = MTNGStackReg.polesDeg7[p];
                double zn = Math.pow(z1, (double)height);
                double sum = (1.0 + MTNGStackReg.polesDeg7[p]) * (target[i]+ zn * target[(height-1)*width + i]);
                zn*=zn;
                for(int n = 1;n < height - 1;n++)
                {
                    z1 *= MTNGStackReg.polesDeg7[p];
                    zn /= MTNGStackReg.polesDeg7[p];
                    sum += (z1 + zn) * target[n*width + i];
                }
                target[i] = (sum / (1.0 - Math.pow(MTNGStackReg.polesDeg7[p], (double)(2*height))));
                // causal recursion
                for(int n = 1;n < height;n++)
                {
                    target[n*width+i] += MTNGStackReg.polesDeg7[p] * target[(n-1)*width+i];
                }
                // anticausal initialization
                target[(height-1)*width + i] = (MTNGStackReg.polesDeg7[p] * target[(height-1)*width + i] / (MTNGStackReg.polesDeg7[p] - 1.0));

                // anticausal recursion
                for(int n = height - 2;n >= 0;n--)
                {
                    target[n*width + i] = MTNGStackReg.polesDeg7[p] * (target[(n+1)*width + i] - target[n*width + i]);
                }
            }
        }
    }
    public static void basicToCardinal2DXhpDeg7(final double input[], final double output[], final int width, final int height)
    {
        /*
         * All the other mirroring conditions can safely be ignored because width > 6 is guaranteed
         * run for all non-border condition pixels to prevent checking the conditions all the time
         * symmetricFirMirrorOffBounds1D
         */
        for(int i = 0;i < height;i++)
        {
            int nIndex = i * width + 3;
            for(int n = 3;n < width-3;n++,nIndex++)
            {
                output[nIndex] = MTNGStackReg.h0D7 * input[nIndex] + MTNGStackReg.h1D7 * (input[nIndex-1] + input[nIndex+1]) + MTNGStackReg.h2D7 * (input[nIndex-2] + input[nIndex+2]) + MTNGStackReg.h3D7 * (input[nIndex-3] + input[nIndex+3]);
            }
            // the left boundary condition
            // n == 0
            nIndex = i * width;
            output[nIndex] = MTNGStackReg.h0D7 * input[nIndex] + MTNGStackReg.h1D7 * (input[nIndex] + input[nIndex+1]) + MTNGStackReg.h2D7 * (input[nIndex+1] + input[nIndex+2])+ MTNGStackReg.h3D7 * (input[nIndex+2] + input[nIndex+3]);
            // n == 1
            nIndex++;
            output[nIndex] = MTNGStackReg.h0D7 * input[nIndex] + MTNGStackReg.h1D7 * (input[nIndex-1] + input[nIndex+1]) + MTNGStackReg.h2D7 * (input[nIndex-1] + input[nIndex+2]) + MTNGStackReg.h3D7 * (input[nIndex] + input[nIndex+3]);
            // n == 2
            nIndex++;
            output[nIndex] = MTNGStackReg.h0D7 * input[nIndex] + MTNGStackReg.h1D7 * (input[nIndex-1] + input[nIndex+1]) + MTNGStackReg.h2D7 * (input[nIndex-2] + input[nIndex+2]) + MTNGStackReg.h3D7 * (input[nIndex-2] + input[nIndex+3]);
            // the right boundary condition
            // n == width - 3
            nIndex = (i+1)*width - 3;
            output[nIndex] = MTNGStackReg.h0D7 * input[nIndex] + MTNGStackReg.h1D7 * (input[nIndex-1] + input[nIndex+1]) + MTNGStackReg.h2D7 * (input[nIndex-2] + input[nIndex+2]) + MTNGStackReg.h3D7 * (input[nIndex-3] + input[nIndex+2]);
            // n == width - 2
            nIndex++;
            output[nIndex] = MTNGStackReg.h0D7 * input[nIndex] + MTNGStackReg.h1D7 * (input[nIndex-1] + input[nIndex+1]) + MTNGStackReg.h2D7 * (input[nIndex-2] + input[nIndex+1]) + MTNGStackReg.h3D7 * (input[nIndex-3] + input[nIndex]);
            // n == width -1
            nIndex++;
            output[nIndex] = MTNGStackReg.h0D7 * input[nIndex] + MTNGStackReg.h1D7 * (input[nIndex-1] + input[nIndex]) + MTNGStackReg.h2D7 * (input[nIndex-2] + input[nIndex-1]) + MTNGStackReg.h3D7 * (input[nIndex-3] + input[nIndex-2]);
        }
    }
    public static void basicToCardinal2DYhpDeg7(final double input[], final double output[], final int width, final int height)
    {
        /*
         * All the other mirroring conditions can safely be ignored because width > 6 is guaranteed
         * run for all non-border condition pixels to prevent checking the conditions all the time
         * symmetricFirMirrorOffBounds1D
         */
        for(int i = 0;i < width;i++)
        {
            int nIndex = 3 * width + i;
            for(int n = 3;n < height-3;n++,nIndex+=width)
            {
                output[nIndex] = MTNGStackReg.h0D7 * input[nIndex] + MTNGStackReg.h1D7 * (input[nIndex-width] + input[nIndex+width]) + MTNGStackReg.h2D7 * (input[nIndex-2*width] + input[nIndex+2*width]) + MTNGStackReg.h3D7 * (input[nIndex-3*width] + input[nIndex+3*width]);
            }
            // the top boundary condition
            // n == 0
            nIndex = i;
            output[nIndex] = MTNGStackReg.h0D7 * input[nIndex] + MTNGStackReg.h1D7 * (input[nIndex] + input[nIndex+width]) + MTNGStackReg.h2D7 * (input[nIndex+width] + input[nIndex+2*width])+ MTNGStackReg.h3D7 * (input[nIndex+2*width] + input[nIndex+3*width]);
            // n == 1
            nIndex+=width;
            output[nIndex] = MTNGStackReg.h0D7 * input[nIndex] + MTNGStackReg.h1D7 * (input[nIndex-width] + input[nIndex+width]) + MTNGStackReg.h2D7 * (input[nIndex-width] + input[nIndex+2*width]) + MTNGStackReg.h3D7 * (input[nIndex] + input[nIndex+3*width]);
            // n == 2
            nIndex+=width;
            output[nIndex] = MTNGStackReg.h0D7 * input[nIndex] + MTNGStackReg.h1D7 * (input[nIndex-width] + input[nIndex+width]) + MTNGStackReg.h2D7 * (input[nIndex-2*width] + input[nIndex+2*width]) + MTNGStackReg.h3D7 * (input[nIndex-2*width] + input[nIndex+3*width]);
            // the bottom boundary condition
            // n == height - 3
            nIndex = (height-3)*width + i;
            output[nIndex] = MTNGStackReg.h0D7 * input[nIndex] + MTNGStackReg.h1D7 * (input[nIndex-width] + input[nIndex+width]) + MTNGStackReg.h2D7 * (input[nIndex-2*width] + input[nIndex+2*width]) + MTNGStackReg.h3D7 * (input[nIndex-3*width] + input[nIndex+2*width]);
            // n == height - 2
            nIndex+=width;
            output[nIndex] = MTNGStackReg.h0D7 * input[nIndex] + MTNGStackReg.h1D7 * (input[nIndex-width] + input[nIndex+width]) + MTNGStackReg.h2D7 * (input[nIndex-2*width] + input[nIndex+width]) + MTNGStackReg.h3D7 * (input[nIndex-3*width] + input[nIndex]);
            // n == height -1
            nIndex+=width;
            output[nIndex] = MTNGStackReg.h0D7 * input[nIndex] + MTNGStackReg.h1D7 * (input[nIndex-width] + input[nIndex]) + MTNGStackReg.h2D7 * (input[nIndex-2*width] + input[nIndex-width]) + MTNGStackReg.h3D7 * (input[nIndex-3*width] + input[nIndex-2*width]);
        }
    }
    public static void reduceDual1DX(final double input[], final double output[], final int width, final int height, final int halfwidth)
    {
        for(int i = 0; i < height;i++)
        {
            int nIndex = i * halfwidth + 1;
            int rIndex = i * width + 2;
            for(int n = 1;n < halfwidth - 1;n++,nIndex++,rIndex+=2)
            {
                output[nIndex] = MTNGStackReg.rh0 * input[rIndex] + MTNGStackReg.rh1 * (input[rIndex - 1] + input[rIndex + 1]) + MTNGStackReg.rh2 * (input[rIndex - 2] + input[rIndex + 2]);
            }
            // the mirror boundary conditions
            // n == 0
            nIndex = i * halfwidth;
            rIndex = i * width;
            output[nIndex] =  MTNGStackReg.rh0 * input[rIndex] + MTNGStackReg.rh1 * (input[rIndex] + input[rIndex+1]) + MTNGStackReg.rh2 * (input[rIndex+1] + input[rIndex+2]);
            // n == halfwidth - 1
            nIndex = (i+1) * halfwidth - 1;
            rIndex = (i+1) * width - 2;
            if(width == (2 * halfwidth))// Yes this can be different if width % 2 != 0
            {
                output[nIndex] = MTNGStackReg.rh0 * input[rIndex] + MTNGStackReg.rh1 * (input[rIndex-1] + input[rIndex+1]) + MTNGStackReg.rh2 * (input[rIndex-2] + input[rIndex+1]);
            }
            else
            {
                output[nIndex] = MTNGStackReg.rh0 * input[rIndex-1] + MTNGStackReg.rh1 * (input[rIndex-2] + input[rIndex]) + MTNGStackReg.rh2 * (input[rIndex-3] + input[rIndex+1]);
            }
        }
    }
    public static void reduceDual1DY(final double input[], final double output[], final int halfwidth, final int height, final int halfheight)
    {
        for(int i = 0; i < halfwidth;i++)
        {
            int nIndex = i + halfwidth;
            int rIndex = i + 2 * halfwidth;
            for(int n = 1;n < halfheight - 1;n++,nIndex+=halfwidth,rIndex+=2*halfwidth)
            {
                output[nIndex] = MTNGStackReg.rh0 * input[rIndex] + MTNGStackReg.rh1 * (input[rIndex - halfwidth] + input[rIndex + halfwidth]) + MTNGStackReg.rh2 * (input[rIndex - 2*halfwidth] + input[rIndex + 2*halfwidth]);
            }
            // the mirror boundary conditions
            // n == 0
            nIndex = i;
            output[nIndex] =  MTNGStackReg.rh0 * input[nIndex] + MTNGStackReg.rh1 * (input[nIndex] + input[nIndex+halfwidth]) + MTNGStackReg.rh2 * (input[nIndex+halfwidth] + input[nIndex+2*halfwidth]);
            // n == halfheight - 1
            nIndex = (halfheight - 1) * halfwidth + i;
            rIndex = (height - 2) * halfwidth + i;
            if(height == (2 * halfheight))// Yes this can be different if height % 2 != 0
            {
                output[nIndex] = MTNGStackReg.rh0 * input[rIndex] + MTNGStackReg.rh1 * (input[rIndex-halfwidth] + input[rIndex+halfwidth]) + MTNGStackReg.rh2 * (input[rIndex-2*halfwidth] + input[rIndex+halfwidth]);
            }
            else
            {
                output[nIndex] = MTNGStackReg.rh0 * input[rIndex-halfwidth] + MTNGStackReg.rh1 * (input[rIndex-2*halfwidth] + input[rIndex]) + MTNGStackReg.rh2 * (input[rIndex-3*halfwidth] + input[rIndex+halfwidth]);
            }
        }
    }
    private void constructTargetImagePyramid()
    {
        // TODO: use localWorkGroup size in a more sensible manner
        int width = (int)sharedContext.img.dimension(0);
        int height = (int)sharedContext.img.dimension(1);
        converter.convertTo(scat.targetArray, targetPyramid[0].Coefficient);
        // pre-multiply the image for cubic spline interpolation
        premultiplyCubicBSpline(targetPyramid[0].Coefficient, width * height);
        // Conversion to B-spline coefficients along X axis
        cubicBSplinePrefilter2DXhp(targetPyramid[0].Coefficient, width, height);
        // pre-multiply again
        premultiplyCubicBSpline(targetPyramid[0].Coefficient, width * height);
        // along the Y-axis
        cubicBSplinePrefilter2DYhp(targetPyramid[0].Coefficient, width, height);

        // Now prepare the image for resampling by applying the FIR filter of degree 7 (out of place mod)
        // X-FIR
        basicToCardinal2DXhpDeg7(targetPyramid[0].Coefficient, entryImageBuffers, width, height);
        // Y-FIR
        basicToCardinal2DYhpDeg7(entryImageBuffers, fullSizedHelperBuffer, width, height);

        // Start the reduction loop
        // reduce in x direction
        reduceDual1DX(fullSizedHelperBuffer, entryImageBuffers, width, height, width/2);
        for(int j = 1;j < pyramidDepth; j++)
        {
            // reduce in y direction
            reduceDual1DY(entryImageBuffers, targetPyramid[j].Coefficient, width/2, height, height/2);
            if(j < pyramidDepth - 1)
            {
                // reduce in x direction
                reduceDual1DX(targetPyramid[j].Coefficient,entryImageBuffers, width/2, height/2, (int)(((int)(width/2))/2));// Warning: integer division don't change
            }
            width /= 2;
            height /= 2;
            // restore the B-spline coefficients
            // pre-multiply
            premultiplyCubicBSplineDeg7(targetPyramid[j].Coefficient,width*height);
            // x-restoration
            cubicBSplinePrefilter2DXhpDeg7(targetPyramid[j].Coefficient, width, height);
            // pre-multiply again
            premultiplyCubicBSplineDeg7(targetPyramid[j].Coefficient,width*height);
            // y-restoration
            cubicBSplinePrefilter2DYhpDeg7(targetPyramid[j].Coefficient, width, height);
        }
    }
    public static void antiSymmetricFirMirrorOffBounds1DX(final double input[], final double output[], final int width, final int height)
    {
        for(int i = 0;i < height;i++)
        {
            int nIndex = i * width + 1;
            for(int n = 1;n < width - 1;n++,nIndex++)
            {
                output[nIndex] = (0.5 * (input[nIndex + 1] - input[nIndex - 1]));
            }
            // n == 0
            nIndex = i * width;
            output[nIndex] = (0.5 * (input[nIndex + 1] - input[nIndex]));

            // n == width - 1
            nIndex += width - 1;
            output[nIndex] = (0.5 * (input[nIndex] - input[nIndex - 1]));
        }
    }
    public static void antiSymmetricFirMirrorOffBounds1DY(final double input[], final double output[], final int width, final int height)
    {
        for(int i = 0;i < width;i++)
        {
            int nIndex = i + width;
            for(int n = 1;n < height - 1;n++,nIndex+=width)
            {
                output[nIndex] = (0.5 * (input[nIndex + width] - input[nIndex - width]));
            }
            // n == 0
            output[i] = (0.5 * (input[i + width] - input[i]));

            // n == width - 1
            nIndex = (height-1)*width + i;
            output[nIndex] = (0.5 * (input[nIndex] - input[nIndex - width]));
        }
    }
    public static void targetedPremultiplyCubicBSpline(final double source[], final double target[], final int nrOfElements)
    {
        // This function only exists to allow for copying the data while actually calculating something
        for(int i = 0;i < nrOfElements;i++)
        {
            target[i] = source[i] * 6.0;
        }
    }
    public static void basicToCardinal2DXhp(final double input[], final double output[], final int width, final int height)
    {
        for(int i = 0;i < height;i++)
        {
            int nIndex = i * width + 1;
            for(int n = 1;n < width - 1;n++, nIndex++)
            {
                output[nIndex] = MTNGStackReg.h0D3 * input[nIndex] + MTNGStackReg.h1D3 * (input[nIndex - 1] + input[nIndex + 1]);
            }
            // n == 0
            nIndex = i * width;
            output[nIndex] = MTNGStackReg.h0D3 * input[nIndex] + MTNGStackReg.h1D3 * (input[nIndex] + input[nIndex+1]);
            // n == width - 1
            nIndex += width - 1;
            output[nIndex] = MTNGStackReg.h0D3 * input[nIndex] + MTNGStackReg.h1D3 * (input[nIndex-1] + input[nIndex]);
        }
    }
    public static void basicToCardinal2DYhp(final double input[], final double output[], final int width, final int height)
    {
        for(int i = 0;i < width;i++)
        {
            int nIndex = i + width;
            for(int n = 1;n < height - 1;n++, nIndex+=width)
            {
                output[nIndex] = MTNGStackReg.h0D3 * input[nIndex] + MTNGStackReg.h1D3 * (input[nIndex - width] + input[nIndex + width]);
            }
            // n == 0
            output[i] = MTNGStackReg.h0D3 * input[i] + MTNGStackReg.h1D3 * (input[i] + input[i+width]);
            // n == width - 1
            nIndex = (height-1)*width + i;
            output[nIndex] = MTNGStackReg.h0D3 * input[nIndex] + MTNGStackReg.h1D3 * (input[nIndex-width] + input[nIndex]);
        }
    }
    private void constructSourceImagePyramid()
    {
        int width = (int)sharedContext.img.dimension(0);
        int height = (int)sharedContext.img.dimension(1);
        converter.convertTo(scat.sourceArray, sourcePyramid[0].Image);
        System.arraycopy(sourcePyramid[0].Image, 0, entryImageBuffers, 0, width*height);
        // Conversion to B-spline coefficients
        // pre-multiply the image for cubic spline interpolation
        premultiplyCubicBSpline(entryImageBuffers, width * height);
        // Conversion to B-spline coefficients along X axis
        cubicBSplinePrefilter2DXhp(entryImageBuffers, width, height);

        // X-derivatives
        antiSymmetricFirMirrorOffBounds1DX(entryImageBuffers, sourcePyramid[0].xGradient, width, height);

        // pre-multiply again
        premultiplyCubicBSpline(entryImageBuffers, width * height);
        // Now along the Y-axis
        cubicBSplinePrefilter2DYhp(entryImageBuffers, width, height);
        // for the source only the images in the pyramid are needed so no need to copy the B-spline coefficients

        // The Y-derivatives still need to be calculated from the Y-coefficients
        // First calculate the Y-coefficients and only the Y-coefficients
        // Has to be pre-multiplied by lambda again, avoid copying data so use an out-of-place modifying calculation
        targetedPremultiplyCubicBSpline(sourcePyramid[0].Image, fullSizedHelperBuffer, width*height);
        cubicBSplinePrefilter2DYhp(fullSizedHelperBuffer, width, height);
        // calculate the derivatives in the Y-direction
        antiSymmetricFirMirrorOffBounds1DY(fullSizedHelperBuffer, sourcePyramid[0].yGradient, width, height);

        // prepare the image for resampling by applying the FIR filter of degree 7 (out of place mod)
        // X-FIR
        basicToCardinal2DXhpDeg7(entryImageBuffers, fullSizedHelperBuffer, width, height);
        // Y-FIR
        basicToCardinal2DYhpDeg7(fullSizedHelperBuffer, entryImageBuffers, width, height);

        // Start the reduction loop
        // reduce in x direction
        reduceDual1DX(entryImageBuffers, fullSizedHelperBuffer, width, height, width/2);
        for(int j = 1;j < pyramidDepth; j++)
        {
            // reduce in y direction
            reduceDual1DY(fullSizedHelperBuffer, sourcePyramid[j].Image, width/2, height, height/2);
            if(j < pyramidDepth - 1)
            {
                // reduce in x direction
                reduceDual1DX(sourcePyramid[j].Image,fullSizedHelperBuffer, width/2, height/2, (int)(((int)(width/2))/2));// Warning: integer division don't change
            }
            width /= 2;
            height /= 2;
            // restore the B-spline coefficients
            // pre-multiply
            premultiplyCubicBSplineDeg7(sourcePyramid[j].Image,width*height);
            // x-restoration
            cubicBSplinePrefilter2DXhpDeg7(sourcePyramid[j].Image, width, height);
            // pre-multiply again
            premultiplyCubicBSplineDeg7(sourcePyramid[j].Image,width*height);
            // y-restoration
            cubicBSplinePrefilter2DYhpDeg7(sourcePyramid[j].Image, width, height);
            // Now the downsampled images need to be restored and the derivatives have to be calculated
            antiSymmetricFirMirrorOffBounds1DX(sourcePyramid[j].Image, entryImageBuffers, width, height);
            // Because all filters are linearly separable the Y coefficients may simply be restored on the X-diff and vice versa
            basicToCardinal2DYhp(entryImageBuffers, sourcePyramid[j].xGradient, width, height);
            // Now Y
            antiSymmetricFirMirrorOffBounds1DY(sourcePyramid[j].Image, entryImageBuffers, width, height);
            basicToCardinal2DXhp(entryImageBuffers, sourcePyramid[j].yGradient, width, height);

            // restore the actual downsampled image from the B-spline coefficients residing in sourcePyramid[j].Image
            basicToCardinal2DXhp(sourcePyramid[j].Image, entryImageBuffers, width, height);
            basicToCardinal2DYhp(entryImageBuffers, sourcePyramid[j].Image, width, height);
        }
    }

    private void doRegistration()
    {
        iterationPower = (int)Math.pow(2.0, (double)pyramidDepth);
        switch (sharedContext.transformationType) {
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
                // scale up (but the rotation and scale are not scale dependent)
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
                // scale up (but the rotation and scale are not scale dependent)
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
            final double deltaKappa = update[0];
            final double scaleFactor = Math.exp(deltaKappa);
            /*
             * Fixed composition of the update: first scale, then rotate, then translate. 
             * It is important to keep this order for the optimization to work properly 
             * (the mean squares are calculated with this order in mind).
             */
            //currentscale = this.scale + update[0];
            currentscale = this.scale * scaleFactor;
            currentangle = this.angle - update[1];
            //displacement = Math.sqrt(update[2] * update[2] + update[3] * update[3]) + 0.25 * Math.sqrt((double)(targetPyramid[pyramidIndex].width * targetPyramid[pyramidIndex].width) + (double)(targetPyramid[pyramidIndex].height * targetPyramid[pyramidIndex].height)) * (Math.abs(update[0]) + Math.abs(update[1]));
            displacement = Math.sqrt(update[2] * update[2] + update[3] * update[3]) + 0.25 * Math.sqrt((double)(targetPyramid[pyramidIndex].width * targetPyramid[pyramidIndex].width) + (double)(targetPyramid[pyramidIndex].height * targetPyramid[pyramidIndex].height)) * (Math.abs(deltaKappa) + Math.abs(update[1]));
            c = Math.cos(update[1]);
            s = Math.sin(update[1]);
            //currentoffsetx = ((offsetx + update[2]) * c - (offsety + update[3]) * s) * (1.0 + update[0]);
            //currentoffsety = ((offsetx + update[2]) * s + (offsety + update[3]) * c) * (1.0 + update[0]);
            final double trialTx = offsetx + update[2];
            final double trialTy = offsety + update[3];
            currentoffsetx = scaleFactor * (trialTx * c - trialTy * s);
            currentoffsety = scaleFactor * (trialTx * s + trialTy * c);
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
        //currentscale = this.scale + update[0];
        final double deltaKappa = update[0];
        final double scaleFactor = Math.exp(deltaKappa);
        currentscale = this.scale * scaleFactor;
        currentangle = this.angle - update[1];
        c = Math.cos(update[1]);
        s = Math.sin(update[1]);
        //currentoffsetx = ((offsetx + update[2]) * c - (offsety + update[3]) * s) * (1.0 + update[0]);
        //currentoffsety = ((offsetx + update[2]) * s + (offsety + update[3]) * c) * (1.0 + update[0]);
        final double trialTx = offsetx + update[2];
        final double trialTy = offsety + update[3];
        currentoffsetx = scaleFactor * (trialTx * c - trialTy * s);
        currentoffsety = scaleFactor * (trialTx * s + trialTy * c);
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
     * Accepted affine step is applied compositionally (group-correct), not component-wise:
     *
     *   B = I + dA, with dA = [[δa11, δa12], [δa21, δa22]]
     *   A_new = B * A
     *   t_new = B * (t + δt)
     *
     * where A = [[a11, a12], [a21, a22]] and t = (tx, ty).
     *
     * This matches the same left-composition convention used by the transformation
     * classes and the stack-combiner workers:
     *   T_new = ΔT ∘ T_current.
     *
     * Contrast with rigid body and scaled rotation:
     * - Rigid body uses rotational composition with angle subtraction in this inverse
     *   Jacobian convention, then composes translation through the incremental rotation.
     * - Scaled rotation uses log-scale increment (κ = log s), multiplicative scale update,
     *   and compositional translation update.
     * Displacement convergence criterion:
     *   displacement = sqrt(δtx² + δty²)
     *                + 0.5*width *(|δa11| + |δa21|)
     *                + 0.5*height*(|δa12| + |δa22|)
     * This is a first-order pixel-motion proxy derived from the maximal |x|,|y| in
     * the image domain (about width/2 and height/2).
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
            /*currenta11    = this.a11    + update[0];
            currenta12    = this.a12    + update[1];
            currenta21    = this.a21    + update[2];
            currenta22    = this.a22    + update[3];
            currentoffsetx = this.offsetx + update[4];
            currentoffsety = this.offsety + update[5];*/
            
            //Group-correct composition of the affine update: A_new = B * A, t_new = B * (t + δt)
            final double b11 = 1.0 + update[0];
            final double b12 = update[1];
            final double b21 = update[2];
            final double b22 = 1.0 + update[3];

            currenta11 = b11 * this.a11 + b12 * this.a21;
            currenta12 = b11 * this.a12 + b12 * this.a22;
            currenta21 = b21 * this.a11 + b22 * this.a21;
            currenta22 = b21 * this.a12 + b22 * this.a22;

            final double trialTx = this.offsetx + update[4];
            final double trialTy = this.offsety + update[5];
            currentoffsetx = b11 * trialTx + b12 * trialTy;
            currentoffsety = b21 * trialTx + b22 * trialTy;

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

        /*currenta11     = this.a11     + update[0];
        currenta12     = this.a12     + update[1];
        currenta21     = this.a21     + update[2];
        currenta22     = this.a22     + update[3];
        currentoffsetx = this.offsetx + update[4];
        currentoffsety = this.offsety + update[5];*/
        final double b11 = 1.0 + update[0];
        final double b12 = update[1];
        final double b21 = update[2];
        final double b22 = 1.0 + update[3];

        currenta11 = b11 * this.a11 + b12 * this.a21;
        currenta12 = b11 * this.a12 + b12 * this.a22;
        currenta21 = b21 * this.a11 + b22 * this.a21;
        currenta22 = b21 * this.a12 + b22 * this.a22;

        final double trialTx = this.offsetx + update[4];
        final double trialTy = this.offsety + update[5];
        currentoffsetx = b11 * trialTx + b12 * trialTy;
        currentoffsety = b21 * trialTx + b22 * trialTy;

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
    
    private double getTranslationMeanSquares(int pyramidIndex, double currentoffsetx, double currentoffsety)
    {
        // First reset the global values which will not be reset in the loop
        Arrays.fill(gradient, 0.0);
        for (int i = 0; (i < 2); i++) {
            Arrays.fill(hessian[i], 0.0);
        }
        
        final int width = (int)sourcePyramid[pyramidIndex].width;
        final int height = (int)sourcePyramid[pyramidIndex].height;
        final double[] source = sourcePyramid[pyramidIndex].Image;
        final double[] xGradient = sourcePyramid[pyramidIndex].xGradient;
        final double[] yGradient = sourcePyramid[pyramidIndex].yGradient;
        final int targetwidth = (int)targetPyramid[pyramidIndex].width;
        final int doubletargetwidth = targetwidth * 2;
        final int targetheight = (int)targetPyramid[pyramidIndex].height;
        final int doubletargetheight = targetheight * 2;
        final double[] target = targetPyramid[pyramidIndex].Coefficient;
        int nIndex = 0;
        int area = 0;
        double s;
        double msqe = 0.0;// Mean square error
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
                if((mskx >= 0)&&(mskx < targetwidth)&&(msky >= 0)&&(msky < targetheight))
                {
                    area++;
                    
                    computeXInterpolationIndices(coordx, doubletargetwidth, targetwidth, xInterpolationIndices);
                    computeYInterpolationIndices(coordy, doubletargetheight, targetheight, targetwidth, yInterpolationIndices);

                    rescoordx = getFractional(coordx);
                    rescoordy = getFractional(coordy);
                    
                    s = interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, target);

                    // now calculate the return value
                    double diff = source[nIndex] - s;
                    msqe += diff * diff;
                    /*
                    TODO/FIXME/KNOWN ISSUE:
                    The following summation is MUCH worse than the parallel sum reduction done on the GPU, because (relatively speaking)
                    small numbers are added to an ever growing larger number, reducing the precision in the outcome. Currently
                    I ignore this like the original implementation, but this is one of many reasons why the GPU version and the CPU version
                    will never yield the same results!
                    */
                    gradient[0] += diff * xGradient[nIndex];
                    gradient[1] += diff * yGradient[nIndex];
                    hessian[0][0] += xGradient[nIndex] * xGradient[nIndex];
                    hessian[0][1] += xGradient[nIndex] * yGradient[nIndex];
                    hessian[1][1] += yGradient[nIndex] * yGradient[nIndex];
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
        return msqe / ((double)area);
    }
    
    private double getTranslationMeanSquaresWithoutHessian(int pyramidIndex, double currentoffsetx, double currentoffsety)
    {
        final int width = (int)sourcePyramid[pyramidIndex].width;
        final int height = (int)sourcePyramid[pyramidIndex].height;
        final double[] source = sourcePyramid[pyramidIndex].Image;
        final int targetwidth = (int)targetPyramid[pyramidIndex].width;
        final int targetheight = (int)targetPyramid[pyramidIndex].height;
        final int doubletargetwidth = targetwidth * 2;
        final int doubletargetheight = targetheight * 2;
        final double[] target = targetPyramid[pyramidIndex].Coefficient;
        int nIndex = 0;
        int area = 0;
        double s;
        double msqe = 0.0;// Mean square error
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
                if((mskx >= 0)&&(mskx < targetwidth)&&(msky >= 0)&&(msky < targetheight))
                {
                    area++;
                    
                    computeXInterpolationIndices(coordx, doubletargetwidth, targetwidth, xInterpolationIndices);
                    computeYInterpolationIndices(coordy, doubletargetheight, targetheight, targetwidth, yInterpolationIndices);

                    rescoordx = getFractional(coordx);
                    rescoordy = getFractional(coordy);
                    
                    s = interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, target);

                    // calculate the return values
                    double diff = source[nIndex] - s;
                    msqe += diff * diff;
                }
                // walk along the X-vector direction
                coordx += 1.0;
            }
        }
        return msqe / ((double)area);
    }

    private double getRigidBodyMeanSquares(int pyramidIndex, double currentoffsetx, double currentoffsety, double currentangle)
    {
        // First reset the global values which will not be reset in the loop
        Arrays.fill(gradient, 0.0);
        for (int i = 0; (i < 3); i++) {
            Arrays.fill(hessian[i], 0.0);
        }
        
        final int width = (int)sourcePyramid[pyramidIndex].width;
        final int height = (int)sourcePyramid[pyramidIndex].height;
        final double[] source = sourcePyramid[pyramidIndex].Image;
        final double[] xGradient = sourcePyramid[pyramidIndex].xGradient;
        final double[] yGradient = sourcePyramid[pyramidIndex].yGradient;
        final int targetwidth = (int)targetPyramid[pyramidIndex].width;
        final int doubletargetwidth = targetwidth * 2;
        final int targetheight = (int)targetPyramid[pyramidIndex].height;
        final int doubletargetheight = targetheight * 2;
        final double[] target = targetPyramid[pyramidIndex].Coefficient;
        int nIndex = 0;
        int area = 0;
        double s;
        double msqe = 0.0;// Mean square error
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
                if((mskx >= 0)&&(mskx < targetwidth)&&(msky >= 0)&&(msky < targetheight))
                {
                    area++;
                    
                    computeXInterpolationIndices(coordx, doubletargetwidth, targetwidth, xInterpolationIndices);
                    computeYInterpolationIndices(coordy, doubletargetheight, targetheight, targetwidth, yInterpolationIndices);

                    rescoordx = getFractional(coordx);
                    rescoordy = getFractional(coordy);
                    
                    s = interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, target);

                    // calculate the return values
                    double diff = source[nIndex] - s;
                    msqe += diff * diff;
                    double theta = yGradient[nIndex] * (double)n - xGradient[nIndex] * (double)i;
                    /*
                    TODO/FIXME/KNOWN ISSUE:
                    The following summation is MUCH worse than the parallel sum reduction done on the GPU, because (relatively speaking)
                    small numbers are added to an ever growing larger number, reducing the precision in the outcome. Currently
                    I ignore this like the original implementation, but this is one of many reasons why the GPU version and the CPU version
                    will never yield the same results!
                    */
                    gradient[0] += diff * theta;
                    gradient[1] += diff * xGradient[nIndex];
                    gradient[2] += diff * yGradient[nIndex];
                    hessian[0][0] += theta * theta;
                    hessian[0][1] += theta * xGradient[nIndex];
                    hessian[0][2] += theta * yGradient[nIndex];
                    hessian[1][1] += xGradient[nIndex] * xGradient[nIndex];
                    hessian[1][2] += xGradient[nIndex] * yGradient[nIndex];
                    hessian[2][2] += yGradient[nIndex] * yGradient[nIndex];
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
        return msqe / ((double)area);
    }
    private double getRigidBodyMeanSquaresWithoutHessian(int pyramidIndex, double currentoffsetx, double currentoffsety, double currentangle)
    {
        final int width = (int)sourcePyramid[pyramidIndex].width;
        final int height = (int)sourcePyramid[pyramidIndex].height;
        final double[] source = sourcePyramid[pyramidIndex].Image;
        final int targetwidth = (int)targetPyramid[pyramidIndex].width;
        final int targetheight = (int)targetPyramid[pyramidIndex].height;
        final int doubletargetwidth = targetwidth * 2;
        final int doubletargetheight = targetheight * 2;
        final double[] target = targetPyramid[pyramidIndex].Coefficient;
        int nIndex = 0;
        int area = 0;
        double s;
        double msqe = 0.0;// Mean square error
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
                if((mskx >= 0)&&(mskx < targetwidth)&&(msky >= 0)&&(msky < targetheight))
                {
                    area++;
                    
                    computeXInterpolationIndices(coordx, doubletargetwidth, targetwidth, xInterpolationIndices);
                    computeYInterpolationIndices(coordy, doubletargetheight, targetheight, targetwidth, yInterpolationIndices);

                    rescoordx = getFractional(coordx);
                    rescoordy = getFractional(coordy);
                    
                    s = interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, target);

                    // calculate the return values
                    double diff = source[nIndex] - s;
                    msqe += diff * diff;
                }
                // walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
        return msqe / ((double)area);
    }
    
    private double getScaledRotationMeanSquares(int pyramidIndex, double currentoffsetx, double currentoffsety, double currentangle, double currentscale)
    {
        // First reset the global values which will not be reset in the loop
        Arrays.fill(gradient, 0.0);
        for (int i = 0; (i < 4); i++) {
            Arrays.fill(hessian[i], 0.0);
        }
        
        final int width = (int)sourcePyramid[pyramidIndex].width;
        final int height = (int)sourcePyramid[pyramidIndex].height;
        final double[] source = sourcePyramid[pyramidIndex].Image;
        final double[] xGradient = sourcePyramid[pyramidIndex].xGradient;
        final double[] yGradient = sourcePyramid[pyramidIndex].yGradient;
        final int targetwidth = (int)targetPyramid[pyramidIndex].width;
        final int doubletargetwidth = targetwidth * 2;
        final int targetheight = (int)targetPyramid[pyramidIndex].height;
        final int doubletargetheight = targetheight * 2;
        final double[] target = targetPyramid[pyramidIndex].Coefficient;
        int nIndex = 0;
        int area = 0;
        double s;
        double msqe = 0.0;// Mean square error
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
                if((mskx >= 0)&&(mskx < targetwidth)&&(msky >= 0)&&(msky < targetheight))
                {
                    area++;
                    
                    computeXInterpolationIndices(coordx, doubletargetwidth, targetwidth, xInterpolationIndices);
                    computeYInterpolationIndices(coordy, doubletargetheight, targetheight, targetwidth, yInterpolationIndices);

                    rescoordx = getFractional(coordx);
                    rescoordy = getFractional(coordy);
                    
                    s = interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, target);

                    // calculate the return values
                    double diff = source[nIndex] - s;
                    msqe += diff * diff;
                    double theta = yGradient[nIndex] * (double)n - xGradient[nIndex] * (double)i;
                    /*
                     * Log-scale parameter (kappa = log s) with multiplicative composition s <- s*exp(dkappa).
                     * The scale Jacobian column in the inverse method is n*fx + i*fy WITHOUT any current-scale
                     * factor: the forward derivative carries s, but the inverse substitution
                     * grad f_T(W) ~= grad f_S * M^{-1} carries 1/s, and the two cancel exactly. This matches the
                     * identity-frame convention used by the rotation column (theta) and the translation columns
                     * (fx, fy), and by the verified rigid-body optimizer. (Previously this multiplied by
                     * currentscale, which double-counted the scale and mis-scaled the kappa step by O(s) for s != 1.)
                     */
                    final double j_logScale = (((double)n) * xGradient[nIndex] + ((double)i) * yGradient[nIndex]);
                    /*
                    TODO/FIXME/KNOWN ISSUE:
                    The following summation is MUCH worse than the parallel sum reduction done on the GPU, because (relatively speaking)
                    small numbers are added to an ever growing larger number, reducing the precision in the outcome. Currently
                    I ignore this like the original implementation, but this is one of many reasons why the GPU version and the CPU version
                    will never yield the same results!
                    */
                    //gradient[0] += diff * j_scale;
                    gradient[0] += diff * j_logScale;
                    gradient[1] += diff * theta;
                    gradient[2] += diff * xGradient[nIndex];
                    gradient[3] += diff * yGradient[nIndex];
                    //hessian[0][0] += j_scale * j_scale;
                    //hessian[0][1] += j_scale * theta;
                    //hessian[0][2] += j_scale * xGradient[nIndex];
                    //hessian[0][3] += j_scale * yGradient[nIndex];
                    hessian[0][0] += j_logScale * j_logScale;
                    hessian[0][1] += j_logScale * theta;
                    hessian[0][2] += j_logScale * xGradient[nIndex];
                    hessian[0][3] += j_logScale * yGradient[nIndex];
                    hessian[1][1] += theta * theta;
                    hessian[1][2] += theta * xGradient[nIndex];
                    hessian[1][3] += theta * yGradient[nIndex];
                    hessian[2][2] += xGradient[nIndex] * xGradient[nIndex];
                    hessian[2][3] += xGradient[nIndex] * yGradient[nIndex];
                    hessian[3][3] += yGradient[nIndex] * yGradient[nIndex];
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
        
        /*
         * Normalize the MSE by the area and the square of the scale. 
         * The original implementation only normalized by area, 
         * but this can lead to issues when the scale is very small 
         * or very large. By also normalizing by the square of the scale, 
         * we ensure that the MSE is comparable across different 
         * scales and that the optimization is more stable.
         */
        //return msqe / ((double)area);
        final double jacobianNorm = Math.max(currentscale * currentscale, 1.0e-12);
        return msqe / (((double) area) * jacobianNorm);
    }
    private double getScaledRotationMeanSquaresWithoutHessian(int pyramidIndex, double currentoffsetx, double currentoffsety, double currentangle, double currentscale)
    {
        final int width = (int)sourcePyramid[pyramidIndex].width;
        final int height = (int)sourcePyramid[pyramidIndex].height;
        final double[] source = sourcePyramid[pyramidIndex].Image;
        final int targetwidth = (int)targetPyramid[pyramidIndex].width;
        final int targetheight = (int)targetPyramid[pyramidIndex].height;
        final int doubletargetwidth = targetwidth * 2;
        final int doubletargetheight = targetheight * 2;
        final double[] target = targetPyramid[pyramidIndex].Coefficient;
        int nIndex = 0;
        int area = 0;
        double s;
        double msqe = 0.0;// Mean square error
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
                if((mskx >= 0)&&(mskx < targetwidth)&&(msky >= 0)&&(msky < targetheight))
                {
                    area++;
                    
                    computeXInterpolationIndices(coordx, doubletargetwidth, targetwidth, xInterpolationIndices);
                    computeYInterpolationIndices(coordy, doubletargetheight, targetheight, targetwidth, yInterpolationIndices);

                    rescoordx = getFractional(coordx);
                    rescoordy = getFractional(coordy);
                    
                    s = interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, target);

                    // calculate the return values
                    double diff = source[nIndex] - s;
                    msqe += diff * diff;
                }
                // walk along the X-vector direction
                coordx += xvecx;
                coordy += xvecy;
            }
        }
        //return msqe / ((double)area);
        final double jacobianNorm = Math.max(currentscale * currentscale, 1.0e-12);
        return msqe / (((double) area) * jacobianNorm);
    }
    

    /*
     * ========================================================================================
     * getAffineMeanSquares
     * ========================================================================================
     *
     * Paper reference: Thévenaz et al., Section III-A, Eq. (3)-(6), specialized to affine.
     *
     * This function computes:
     *   1. The mean squared error E(p) = (1/|Ω|) Σ [f(x_i) - g(T_p(x_i))]²
     *   2. The gradient vector g_k = Σ r_i * J_ik  (k = 0..5)
     *   3. The Gauss-Newton Hessian H_kl = Σ J_ik * J_il  (upper triangle, then symmetrized)
     *
     * The affine transformation is:
     *   x' = a11*n + a12*i + tx
     *   y' = a21*n + a22*i + ty
     *
     * where (n, i) is the output pixel position (column, row).
     *
     * The "inverse" approach (Section III-A) uses SOURCE image gradients ∇f(x)
     * instead of target gradients ∇g(T(x)). This is the key innovation: since f
     * is fixed, ∇f is computed once during pyramid construction and reused across
     * all iterations. The Hessian H = J^T J is therefore approximately constant,
     * changing only because the set of valid (in-bounds) pixels varies with the
     * transformation parameters.
     *
     * Jacobian derivation for affine:
     *   The parameters are p = (a11, a12, a21, a22, tx, ty).
     *   The coordinate mapping is: [x', y']^T = A * [n, i]^T + [tx, ty]^T
     *
     *   ∂x'/∂a11 = n,    ∂y'/∂a11 = 0
     *   ∂x'/∂a12 = i,    ∂y'/∂a12 = 0
     *   ∂x'/∂a21 = 0,    ∂y'/∂a21 = n
     *   ∂x'/∂a22 = 0,    ∂y'/∂a22 = i
     *   ∂x'/∂tx  = 1,    ∂y'/∂tx  = 0
     *   ∂x'/∂ty  = 0,    ∂y'/∂ty  = 1
     *
     *   By the chain rule (using source gradients for the "inverse" approach):
     *     ∂f/∂a11 = (∂f/∂x)(∂x'/∂a11) + (∂f/∂y)(∂y'/∂a11) = n * ∂f/∂x
     *     ∂f/∂a12 = i * ∂f/∂x
     *     ∂f/∂a21 = n * ∂f/∂y
     *     ∂f/∂a22 = i * ∂f/∂y
     *     ∂f/∂tx  = ∂f/∂x
     *     ∂f/∂ty  = ∂f/∂y
     *
     *   So the Jacobian row for pixel (n, i) is:
     *     J_i = [n*∂f/∂x, i*∂f/∂x, n*∂f/∂y, i*∂f/∂y, ∂f/∂x, ∂f/∂y]
     *
     * This yields 6 gradient entries and 21 unique Hessian entries (upper triangle
     * of the 6×6 symmetric matrix), which are then symmetrized.
     *
     * Note on coordinate system:
     *   The source image and its gradients are indexed by the output pixel position
     *   (n, i) where n is the column (x) and i is the row (y). The coordinate mapping
     *   computes where in the TARGET coefficient array to sample from, using cubic
     *   B-spline interpolation with mirror boundary conditions.
     *
     * The B-spline interpolation, boundary handling, and weight computation are
     * identical to getTranslationMeanSquares and getRigidBodyMeanSquares.
     */
    private double getAffineMeanSquares(int pyramidIndex, double currentoffsetx, double currentoffsety,
                                        double currenta11, double currenta12,
                                        double currenta21, double currenta22)
    {
        // Reset gradient vector (6 entries)
        Arrays.fill(gradient, 0.0);
        for (int i = 0; i < 6; i++) {
            Arrays.fill(hessian[i], 0.0);
        }
        
        final int width = (int)sourcePyramid[pyramidIndex].width;
        final int height = (int)sourcePyramid[pyramidIndex].height;
        final double[] source = sourcePyramid[pyramidIndex].Image;
        final double[] xGradient = sourcePyramid[pyramidIndex].xGradient;
        final double[] yGradient = sourcePyramid[pyramidIndex].yGradient;
        final int targetwidth = (int)targetPyramid[pyramidIndex].width;
        final int doubletargetwidth = targetwidth * 2;
        final int targetheight = (int)targetPyramid[pyramidIndex].height;
        final int doubletargetheight = targetheight * 2;
        final double[] target = targetPyramid[pyramidIndex].Coefficient;
        int nIndex = 0;
        int area = 0;
        double s;
        double msqe = 0.0;
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

        for(int i = 0; i < height; i++)
        {
            /*
             * Reset x-coordinate to the start of this row.
             * coordx = tx + i * a12  (the n=0 position for row i)
             * coordy = ty + i * a22
             */
            coordx = currentoffsetx + ((double)i) * yvecx;
            coordy = currentoffsety + ((double)i) * yvecy;

            for(int n = 0; n < width; n++, nIndex++)
            {
                mskx = (int)Math.round(coordx);
                msky = (int)Math.round(coordy);
                if((mskx >= 0) && (mskx < targetwidth) && (msky >= 0) && (msky < targetheight))
                {
                    // ---- B-spline interpolation (identical to translation/rigid body) ----
                    computeXInterpolationIndices(coordx, doubletargetwidth, targetwidth, xInterpolationIndices);
                    computeYInterpolationIndices(coordy, doubletargetheight, targetheight, targetwidth, yInterpolationIndices);

                    rescoordx = getFractional(coordx);
                    rescoordy = getFractional(coordy);
                    
                    s = interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, target);

                    // ---- Gradient and Hessian accumulation ----

                    /*
                     * Residual: r_i = f(x_i) - g(T_p(x_i))
                     * where f is the source image and g is the B-spline-interpolated target.
                     */
                    double diff = source[nIndex] - s;
                    msqe += diff * diff;
                    area++;

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
                // Advance along the x-direction vector (column step)
                coordx += xvecx;
                coordy += xvecy;
            }
        }

        /*
         * Symmetrize the Hessian: H[j][k] = H[k][j] for j > k.
         * The Gauss-Newton Hessian J^T J is symmetric by construction;
         * we only accumulated the upper triangle for efficiency.
         */
        for (int i = 1; (i < 6); i++) {
            for (int j = 0; (j < i); j++) {
                hessian[i][j] = hessian[j][i];
            }
        }
        //return msqe / ((double) area);
        final double detA = Math.abs(currenta11 * currenta22 - currenta12 * currenta21);
        final double jacobianNorm = Math.max(detA, 1.0e-12);
        return msqe / (((double) area) * jacobianNorm);
    }

    /*
     * ========================================================================================
     * getAffineMeanSquaresWithoutHessian
     * ========================================================================================
     *
     * Paper reference: Same as getAffineMeanSquares, but this is the "accelerated" variant
     * described in Section III-A of Thévenaz et al.
     *
     * This function computes ONLY the mean squared error E(p), WITHOUT accumulating the
     * gradient or Hessian. It is used for the FINAL undamped Gauss-Newton step after the
     * LM loop converges.
     *
     * Rationale: The final step uses the gradient and Hessian from the LAST accepted point
     * (already stored in gradient[] and hessian[]). We only need to evaluate the MSE at
     * the trial point to decide whether to accept the step. Skipping the gradient/Hessian
     * accumulation saves computation.
     *
     * The coordinate mapping, B-spline interpolation, and boundary handling are identical
     * to getAffineMeanSquares. The only difference is that no gradient[] or hessian[]
     * entries are written.
     */
    private double getAffineMeanSquaresWithoutHessian(int pyramidIndex, double currentoffsetx, double currentoffsety,
                                                       double currenta11, double currenta12,
                                                       double currenta21, double currenta22)
    {
        final int width = (int)sourcePyramid[pyramidIndex].width;
        final int height = (int)sourcePyramid[pyramidIndex].height;
        final double[] source = sourcePyramid[pyramidIndex].Image;
        final int targetwidth = (int)targetPyramid[pyramidIndex].width;
        final int doubletargetwidth = targetwidth * 2;
        final int targetheight = (int)targetPyramid[pyramidIndex].height;
        final int doubletargetheight = targetheight * 2;
        final double[] target = targetPyramid[pyramidIndex].Coefficient;
        int nIndex = 0;
        int area = 0;
        double s;
        double msqe = 0.0;
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

        for(int i = 0; i < height; i++)
        {
            coordx = currentoffsetx + ((double)i) * yvecx;
            coordy = currentoffsety + ((double)i) * yvecy;

            for(int n = 0; n < width; n++, nIndex++)
            {
                mskx = (int)Math.round(coordx);
                msky = (int)Math.round(coordy);
                if((mskx >= 0) && (mskx < targetwidth) && (msky >= 0) && (msky < targetheight))
                {
                    computeXInterpolationIndices(coordx, doubletargetwidth, targetwidth, xInterpolationIndices);
                    computeYInterpolationIndices(coordy, doubletargetheight, targetheight, targetwidth, yInterpolationIndices);

                    rescoordx = getFractional(coordx);
                    rescoordy = getFractional(coordy);
                    
                    s = interpolateCubicBSpline(rescoordx, rescoordy, xInterpolationIndices, yInterpolationIndices, target);

                    // Only compute the residual and MSE — no gradient/Hessian
                    double diff = source[nIndex] - s;
                    msqe += diff * diff;
                    area++;
                }
                coordx += xvecx;
                coordy += xvecy;
            }
        }
        //return msqe / ((double) area);
        final double detA = Math.abs(currenta11 * currenta22 - currenta12 * currenta21);
        final double jacobianNorm = Math.max(detA, 1.0e-12);
        return msqe / (((double) area) * jacobianNorm);
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
    static void computeXInterpolationIndices(double coord, int doubletargetwidth, int targetwidth, int[] indices)
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
    static void computeYInterpolationIndices(double coord, int doubletargetheight, int targetheight, int targetwidth, int[] indices)
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
    static double getFractional(double coord)
    {
        return coord - (coord >= 0.0 ? ((double)((int)coord)) : ((double)(((int)coord) - 1)));
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
    static double interpolateCubicBSpline(
            double rescoordx, double rescoordy,
            int[] xInterpolationIndices, int[] yInterpolationIndices,
            double[] target)
    {
        // --- X weights ---
        double sx = 1.0 - rescoordx;
        double xw3 = (sx * sx * sx) / 6.0;
        double sx2 = rescoordx * rescoordx;
        double xw2 = (2.0 / 3.0) - 0.5 * sx2 * (2.0 - rescoordx);
        double xw0 = sx2 * rescoordx / 6.0;
        double xw1 = 1.0 - xw0 - xw2 - xw3;

        // --- Y weights ---
        double sy = 1.0 - rescoordy;
        double yw3 = (sy * sy * sy) / 6.0;
        double sy2 = rescoordy * rescoordy;
        double yw2 = (2.0 / 3.0) - 0.5 * sy2 * (2.0 - rescoordy);
        double yw0 = sy2 * rescoordy / 6.0;
        double yw1 = 1.0 - yw0 - yw2 - yw3;

        double[] xWeightsLocal = { xw0, xw1, xw2, xw3 };
        double[] yWeightsLocal = { yw0, yw1, yw2, yw3 };

        // --- Separable 4x4 interpolation ---
        double s = 0.0;
        for (int y = 0; y < 4; y++)
        {
            int tmpindex = yInterpolationIndices[y];
            double row = 0.0;
            for (int x = 0; x < 4; x++)
            {
                row += xWeightsLocal[x] * target[tmpindex + xInterpolationIndices[x]];
            }
            s += yWeightsLocal[y] * row;
        }
        return s;
    }
}
