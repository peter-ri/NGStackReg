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

#define MODULO2(num) (num & 1)

#ifdef USE_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// only when this is defined expose the double precision code, otherwise it will not be used (like for example on a GPU integrated on an Intel CPU)
__constant const double dLambda = 6.0;
__constant const double dPole = -0.26794919243112270647255365849413;
__constant const double dOne = 1.0;
__constant const double dTwo = 2.0;
__constant const double dh0D3 = 0.66666666666666666666666666666666666666666666666666666666666666667;
__constant const double dh1D3 = 0.16666666666666666666666666666666666666666666666666666666666666667;

__constant const double dh = 0.5;
__constant const double dZero = 0.0;
__constant const double dThree = 3.0;

__kernel void ConvertDoubleToFloat(__global const double *source, __global float *target, const int size)
{
    __private int nIndex = get_global_id(0); // this directly corresponds to the offset
    if(nIndex >= size)
    {
        return;
    }
    target[nIndex] = (float)(source[nIndex]); // There is a bug in my version of the NVidia driver preventing me from using convert_float_sat_rte
}
__kernel void dCubicBSplinePrefilter2Dpremulhp(__global double *image, const int size)
{
    __private int nIndex = get_global_id(0); // this directly corresponds to the offset
    if(nIndex < size)
    {
        image[nIndex] *= dLambda;
    }
    
}
__kernel void dTargetedCubicBSplinePrefilter2Dpremulhp(__global double *image, __global double *target, const int size)
{
    __private int nIndex = get_global_id(0); // this directly corresponds to the offset
    if(nIndex < size)
    {
        target[nIndex] = image[nIndex] * dLambda;
    }
}
__kernel void dCubicBSplinePrefilter2DXhp(__global double *image, const int width, const int height)
{
    // high precision but slower
    __private int nIndex = get_global_id(0); // this directly corresponds to the row!!!
    if(nIndex < height)
    {
        __global double *prow = image + (nIndex * width);
    
        // causal initialization
        __private double z1 = dPole;
        __private double zn = pown(dPole,width);
        __private double Sum = (dOne + dPole) * ( prow[0] + zn * prow[width - 1]);
        zn *= zn;
        for(int k = 1;k < width - 1; k++)
        {
            z1 *= dPole;
            zn /= dPole;
            Sum += (z1 + zn) * prow[k];
        }
        prow[0] = (Sum / (dOne - pown(dPole, 2 * width)));

        // Causal recursion
        for(int k = 1; k < width; k++)
        {
            prow[k] += dPole *  prow[k-1];
        }
        // anticausal initialization
        prow[width - 1] = (dPole * prow[width - 1] / (dPole - dOne));

        // Anticausal recursion
        for(int k = width - 2; 0 <= k; k--)
        {
            prow[k] = dPole * (prow[k+1] -  prow[k]);
        }
    }
}

__kernel void dCubicBSplinePrefilter2DYhp(__global double *image, const int width, const int height)
{
    // high precision but slower
    __private int nIndex = get_global_id(0);//this directly corresponds to the column!!!
    if(nIndex < width)
    {
        __global double *prow = image + nIndex;
    
        // causal initialization
        __private double z1 = dPole;
        __private double zn = pown(dPole,height);
        __private double Sum = (dOne + dPole) * ( prow[0] + zn *  prow[(height - 1)*width]);
        zn *= zn;
        for(int k = 1;k < height - 1; k++)
        {
            z1 *= dPole;
            zn /= dPole;
            Sum += (z1 + zn) *  prow[k * width];
        }
        prow[0] = (Sum / (dOne - pown(dPole, 2 * height)));

        // Causal recursion
        for(int k = 1; k < height; k++)
        {
            prow[k * width] += dPole *  prow[(k-1)*width];
        }

        // anticausal initialization
        prow[(height - 1)*width] = (dPole * prow[(height - 1)*width] / (dPole - dOne));

        // Anticausal recursion
        for(int k = height - 2; 0 <= k; k--)
        {
            prow[k*width] = (dPole * ( prow[(k+1)*width] -  prow[k*width]));
        }
    }    
}

__kernel void dantiSymmetricFirMirrorOffBounds1DX(__global double *image ,__global double *target , const int width, const int height)
{
    __private int nIndex = get_global_id(0); // this directly corresponds to the linear address of the !TARGET! pixel
    if(nIndex < height * width)
    {
        __private int column = nIndex % width;
        __private int row = (nIndex - column)/width;
        if(column > 0 && column < (width - 1))
        {
            // most common case
            target[nIndex] = dh * (image[nIndex + 1] - image[nIndex - 1]);
        }
        else if(column == width - 1)
        {
            target[nIndex] = dh * (image[nIndex] - image[nIndex - 1]);
        }
        else
        {
            // column == 0
            target[nIndex] = dh * (image[nIndex + 1] - image[nIndex]);
        }
    }
}

__kernel void dantiSymmetricFirMirrorOffBounds1DY(__global double *image ,__global double *target , const int width, const int height)
{
    __private int nIndex = get_global_id(0); // this directly corresponds to the linear address of the !TARGET! pixel
    if(nIndex < height * width)
    {
        __private int column = nIndex % width;
        __private int row = (nIndex - column)/width;
        if(row > 0 && row < (height - 1))
        {
            // most common case
            target[nIndex] = dh * (image[(row+1)*width+column] - image[(row-1)*width+column]);
        }
        else if(row == height - 1)
        {
            target[nIndex] = dh * (image[nIndex] - image[(row-1)*width+column]);
        }
        else
        {
            // row == 0
            target[nIndex] = dh * (image[width+column] - image[nIndex]);
        }
    }
}

__kernel void drigidBodyError(const __global double *source ,const __global double *target, __global double *diffout, __global double *mask, const int sourcewidth, const int sourceheight, const int targetwidth, const int targetheight, const double offsetx, const double offsety, const double angle)
{
    __private int nIndex = get_global_id(0); // this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private double2 xvec = (double2)(cos(angle),-sin(angle)); // Warning: this is not the x vector but it is the vector added in the x direction
        __private double2 yvec = (double2)(-xvec.y,xvec.x); // Warning: this is not the y vector but it is the vector added in the y direction
        __private double2 coord = (double2)(offsetx, offsety) + ((double)column) * xvec + ((double)row) * yvec;

        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        __private int doubletargetwidth = 2*targetwidth;
        __private int doubletargetheight = 2*targetheight;
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            mask[nIndex] = dOne;
            // Following is the calculation using mirrored boundaries of the x and y indices of the coefficients used for interpolation
            // calculate the x coordinates for interpolation (loop unwrapped) for speed
            __private int4 xInterpolationIndices;
            __private int p = (coord.x >= 0) ? (((int)trunc(coord.x)) + 2) : (((int)trunc(coord.x)) + 1);
            // loop iteration 0
            __private int q = (p < 0) ? (-1 - p) : p;
            // kick out divergence (but calculating the modulo may actually be slower than divergence so maybe use a slightly diverging statement)
            q = q<doubletargetwidth?q:q%doubletargetwidth;
            // q %= doubletargetwidth; //will allways give the right answer
            /*
            if(q >= doubletargetwidth)
            {
                //q -= (2*targetwidth) * (q / (2*targetwidth)); //Warning: this is an integer division, it doesn't yield q (in fact it is a simple modulo operation)
                
            }
            */
            xInterpolationIndices.x = q >= targetwidth ? (doubletargetwidth - 1 - q) : q;
            //loop iteration 1
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetwidth?q:q%doubletargetwidth;
            xInterpolationIndices.y = q >= targetwidth ? (doubletargetwidth - 1 - q) : q;
            //loop iteration 2
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetwidth?q:q%doubletargetwidth;
            xInterpolationIndices.z = q >= targetwidth ? (doubletargetwidth - 1 - q) : q;
            //loop iteration 3
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetwidth?q:q%doubletargetwidth;
            xInterpolationIndices.w = q >= targetwidth ? (doubletargetwidth - 1 - q) : q;

            __private int4 yInterpolationIndices;
            p = (coord.y >= 0) ? (((int)trunc(coord.y)) + 2) : (((int)trunc(coord.y)) + 1);
            // loop iteration 0
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.x = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth); // this is the linear absolute index NOT the row
            // loop iteration 1
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.y = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth); // this is the linear absolute index NOT the row
            // loop iteration 2
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.z = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth); // this is the linear absolute index NOT the row
            // loop iteration 3
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.w = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth); // this is the linear absolute index NOT the row

            coord.x -= (coord.x >= dZero) ? (double)((int)trunc(coord.x)) : (double)(((int)trunc(coord.x)) - 1); // get the residual should also be possible with trunc
            coord.y -= (coord.y >= dZero) ? (double)((int)trunc(coord.y)) : (double)(((int)trunc(coord.y)) - 1); // get the residual should also be possible with trunc

            // Calculate the weights for interpolation
            __private double4 xWeights;
            __private double s = dOne - coord.x;
            xWeights.w = pown(s,3) / dLambda;
            s = coord.x * coord.x;
            xWeights.z = dTwo / dThree - dh * s * (dTwo - coord.x);
            xWeights.x = s * coord.x / dLambda;
            xWeights.y = dOne - xWeights.x - xWeights.z - xWeights.w;
            __private double4 yWeights;
            s = dOne - coord.y;
            yWeights.w = pown(s,3) / dLambda;
            s = coord.y * coord.y;
            yWeights.z = dTwo / dThree - dh * s * (dTwo - coord.y);
            yWeights.x = s * coord.y / dLambda;
            yWeights.y = dOne - yWeights.x - yWeights.z - yWeights.w;

            // unwrapped interpolation loop (sorry 16 iterations) 4x4 interpolation coefficients
            // y loop 0
            s = yWeights.x * (xWeights.x * target[yInterpolationIndices.x + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.x + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.x + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.x + xInterpolationIndices.w]);
            // y loop 1
            s += yWeights.y * (xWeights.x * target[yInterpolationIndices.y + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.y + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.y + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.y + xInterpolationIndices.w]);
            // y loop 2
            s += yWeights.z * (xWeights.x * target[yInterpolationIndices.z + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.z + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.z + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.z + xInterpolationIndices.w]);
            // y loop 3
            s += yWeights.w * (xWeights.x * target[yInterpolationIndices.w + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.w + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.w + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.w + xInterpolationIndices.w]);
            // now s is the value
            diffout[nIndex] = pown(source[nIndex] - s,2);
        }
        else
        {
            diffout[nIndex] = dZero;
            mask[nIndex] = dZero;
        } 
    }   
}


__kernel void dtranslationError(const __global double *source ,const __global double *target, __global double *diffout, __global double *mask, const int sourcewidth, const int sourceheight, const int targetwidth, const int targetheight, const double offsetx, const double offsety)
{
    __private int nIndex = get_global_id(0); // this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private double2 coord = (double2)(offsetx + ((double)column), offsety + ((double)row));

        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        __private int doubletargetwidth = 2*targetwidth;
        __private int doubletargetheight = 2*targetheight;
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            mask[nIndex] = dOne;
            // Following is the calculation using mirrored boundaries of the x and y indices of the coefficients used for interpolation
            // calculate the x coordinates for interpolation (loop unwrapped) for speed
            __private int4 xInterpolationIndices;
            __private int p = (coord.x >= 0) ? (((int)trunc(coord.x)) + 2) : (((int)trunc(coord.x)) + 1);
            // loop iteration 0
            __private int q = (p < 0) ? (-1 - p) : p;
            // kick out divergence (but calculating the modulo may actually be slower than divergence so maybe use a slightly diverging statement)
            q = q<doubletargetwidth?q:q%doubletargetwidth;
            // q %= doubletargetwidth; //will allways give the right answer
            /*
            if(q >= doubletargetwidth)
            {
                //q -= (2*targetwidth) * (q / (2*targetwidth)); //Warning: this is an integer division, it doesn't yield q (in fact it is a simple modulo operation)
                
            }
            */
            xInterpolationIndices.x = q >= targetwidth ? (doubletargetwidth - 1 - q) : q;
            // loop iteration 1
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetwidth?q:q%doubletargetwidth;
            xInterpolationIndices.y = q >= targetwidth ? (doubletargetwidth - 1 - q) : q;
            // loop iteration 2
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetwidth?q:q%doubletargetwidth;
            xInterpolationIndices.z = q >= targetwidth ? (doubletargetwidth - 1 - q) : q;
            // loop iteration 3
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetwidth?q:q%doubletargetwidth;
            xInterpolationIndices.w = q >= targetwidth ? (doubletargetwidth - 1 - q) : q;

            __private int4 yInterpolationIndices;
            p = (coord.y >= 0) ? (((int)trunc(coord.y)) + 2) : (((int)trunc(coord.y)) + 1);
            // loop iteration 0
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.x = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth); // this is the linear absolute index NOT the row
            // loop iteration 1
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.y = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth); // this is the linear absolute index NOT the row
            // loop iteration 2
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.z = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth); // this is the linear absolute index NOT the row
            // loop iteration 3
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.w = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth); // this is the linear absolute index NOT the row

            coord.x -= (coord.x >= dZero) ? (double)((int)trunc(coord.x)) : (double)(((int)trunc(coord.x)) - 1); // get the residual should also be possible with trunc
            coord.y -= (coord.y >= dZero) ? (double)((int)trunc(coord.y)) : (double)(((int)trunc(coord.y)) - 1); // get the residual should also be possible with trunc

            // Calculate the weights for interpolation
            __private double4 xWeights;
            __private double s = dOne - coord.x;
            xWeights.w = pown(s,3) / dLambda;
            s = coord.x * coord.x;
            xWeights.z = dTwo / dThree - dh * s * (dTwo - coord.x);
            xWeights.x = s * coord.x / dLambda;
            xWeights.y = dOne - xWeights.x - xWeights.z - xWeights.w;
            __private double4 yWeights;
            s = dOne - coord.y;
            yWeights.w = pown(s,3) / dLambda;
            s = coord.y * coord.y;
            yWeights.z = dTwo / dThree - dh * s * (dTwo - coord.y);
            yWeights.x = s * coord.y / dLambda;
            yWeights.y = dOne - yWeights.x - yWeights.z - yWeights.w;

            // unwrapped interpolation loop (sorry 16 iterations) 4x4 interpolation coefficients
            // y loop 0
            s = yWeights.x * (xWeights.x * target[yInterpolationIndices.x + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.x + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.x + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.x + xInterpolationIndices.w]);
            // y loop 1
            s += yWeights.y * (xWeights.x * target[yInterpolationIndices.y + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.y + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.y + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.y + xInterpolationIndices.w]);
            // y loop 2
            s += yWeights.z * (xWeights.x * target[yInterpolationIndices.z + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.z + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.z + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.z + xInterpolationIndices.w]);
            // y loop 3
            s += yWeights.w * (xWeights.x * target[yInterpolationIndices.w + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.w + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.w + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.w + xInterpolationIndices.w]);
            // now s is the value
            diffout[nIndex] = pown(source[nIndex] - s,2);
        }
        else
        {
            diffout[nIndex] = dZero;
            mask[nIndex] = dZero;
        } 
    }   
}

__kernel void drigidBodyErrorWithGradAndHess(const __global double *source ,const __global double *target,const __global double *xGradient,const __global double *yGradient,__global double *grad0,__global double *grad1,__global double *grad2,__global double *hessian00,__global double *hessian01,__global double *hessian02,__global double *hessian11,__global double *hessian12,__global double *hessian22, __global double *diffout, __global double *mask, const int sourcewidth, const int sourceheight, const int targetwidth, const int targetheight, const double offsetx, const double offsety, const double angle)
{
    __private int nIndex = get_global_id(0); // this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private double2 xvec = (double2)(cos(angle),-sin(angle)); // Warning: this is not the x vector but it is the vector added in the x direction
        __private double2 yvec = (double2)(-xvec.y,xvec.x); // Warning: this is not the y vector but it is the vector added in the y direction
        __private double2 coord = (double2)(offsetx, offsety) + ((double)column) * xvec + ((double)row) * yvec;

        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        __private int doubletargetwidth = 2*targetwidth;
        __private int doubletargetheight = 2*targetheight;
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            mask[nIndex] = dOne;
            // Following is the calculation using mirrored boundaries of the x and y indices of the coefficients used for interpolation
            // calculate the x coordinates for interpolation (loop unwrapped) for speed
            __private int4 xInterpolationIndices;
            __private int p = (coord.x >= 0) ? (((int)trunc(coord.x)) + 2) : (((int)trunc(coord.x)) + 1);
            // loop iteration 0
            __private int q = (p < 0) ? (-1 - p) : p;
            q = q<targetwidth?q:q%targetwidth;
            xInterpolationIndices.x = q >= targetwidth ? (targetwidth - 1 - q) : q;
            // loop iteration 1
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<targetwidth?q:q%targetwidth;
            xInterpolationIndices.y = q >= targetwidth ? (targetwidth - 1 - q) : q;
            // loop iteration 2
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<targetwidth?q:q%targetwidth;
            xInterpolationIndices.z = q >= targetwidth ? (targetwidth - 1 - q) : q;
            // loop iteration 3
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<targetwidth?q:q%targetwidth;
            xInterpolationIndices.w = q >= targetwidth ? (targetwidth - 1 - q) : q;

            __private int4 yInterpolationIndices;
            p = (coord.y >= 0) ? (((int)trunc(coord.y)) + 2) : (((int)trunc(coord.y)) + 1);
            // loop iteration 0
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.x = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth); // this is the linear absolute index NOT the row
            // loop iteration 1
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.y = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth); // this is the linear absolute index NOT the row
            // loop iteration 2
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.z = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth); // this is the linear absolute index NOT the row
            // loop iteration 3
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.w = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth); // this is the linear absolute index NOT the row

            coord.x -= (coord.x >= dZero) ? (double)((int)trunc(coord.x)) : (double)(((int)trunc(coord.x)) - 1); // get the residual should also be possible with floor
            coord.y -= (coord.y >= dZero) ? (double)((int)trunc(coord.y)) : (double)(((int)trunc(coord.y)) - 1); // get the residual should also be possible with floor

            // Calculate the weights for interpolation
            __private double4 xWeights;
            __private double s = dOne - coord.x;
            xWeights.w = pown(s,3) / dLambda;
            s = coord.x * coord.x;
            xWeights.z = dTwo / dThree - dh * s * (dTwo - coord.x);
            xWeights.x = s * coord.x / dLambda;
            xWeights.y = dOne - xWeights.x - xWeights.z - xWeights.w;
            __private double4 yWeights;
            s = dOne - coord.y;
            yWeights.w = pown(s,3) / dLambda;
            s = coord.y * coord.y;
            yWeights.z = dTwo / dThree - dh * s * (dTwo - coord.y);
            yWeights.x = s * coord.y / dLambda;
            yWeights.y = dOne - yWeights.x - yWeights.z - yWeights.w;

            // unwrapped interpolation loop (sorry 16 iterations) 4x4 interpolation coefficients
            // y loop 0
            s = yWeights.x * (xWeights.x * target[yInterpolationIndices.x + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.x + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.x + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.x + xInterpolationIndices.w]);
            // y loop 1
            s += yWeights.y * (xWeights.x * target[yInterpolationIndices.y + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.y + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.y + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.y + xInterpolationIndices.w]);
            // y loop 2
            s += yWeights.z * (xWeights.x * target[yInterpolationIndices.z + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.z + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.z + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.z + xInterpolationIndices.w]);
            // y loop 3
            s += yWeights.w * (xWeights.x * target[yInterpolationIndices.w + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.w + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.w + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.w + xInterpolationIndices.w]);
            // now s is the value
            __private double diff = source[nIndex] - s;
            diffout[nIndex] = pown(diff,2);
            __private double Theta = yGradient[nIndex] * (double)column - xGradient[nIndex] * (double)row;
            grad0[nIndex] = diff * Theta;
            grad1[nIndex] = diff * xGradient[nIndex];
            grad2[nIndex] = diff * yGradient[nIndex];
            hessian00[nIndex] = pown(Theta,2);
            hessian01[nIndex] = Theta * xGradient[nIndex];
            hessian02[nIndex] = Theta * yGradient[nIndex];
            hessian11[nIndex] = pown(xGradient[nIndex],2);
            hessian12[nIndex] = xGradient[nIndex] * yGradient[nIndex];
            hessian22[nIndex] = pown(yGradient[nIndex],2);
        }
        else
        {
            grad0[nIndex] = dZero;
            grad1[nIndex] = dZero;
            grad2[nIndex] = dZero;
            hessian00[nIndex] = dZero;
            hessian01[nIndex] = dZero;
            hessian02[nIndex] = dZero;
            hessian11[nIndex] = dZero;
            hessian12[nIndex] = dZero;
            hessian22[nIndex] = dZero;
            diffout[nIndex] = dZero;
            mask[nIndex] = dZero;
        }
    }
}

__kernel void dtranslationErrorWithGradAndHess(const __global double *source ,const __global double *target,const __global double *xGradient,const __global double *yGradient,__global double *grad0,__global double *grad1,__global double *hessian00,__global double *hessian01,__global double *hessian11, __global double *diffout, __global double *mask, const int sourcewidth, const int sourceheight, const int targetwidth, const int targetheight, const double offsetx, const double offsety)
{
    __private int nIndex = get_global_id(0); // this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private double2 coord = (double2)(offsetx + ((double)column), offsety + ((double)row));

        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        __private int doubletargetwidth = 2*targetwidth;
        __private int doubletargetheight = 2*targetheight;
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            mask[nIndex] = dOne;
            // Following is the calculation using mirrored boundaries of the x and y indices of the coefficients used for interpolation
            // calculate the x coordinates for interpolation (loop unwrapped) for speed
            __private int4 xInterpolationIndices;
            __private int p = (coord.x >= 0) ? (((int)trunc(coord.x)) + 2) : (((int)trunc(coord.x)) + 1);
            // loop iteration 0
            __private int q = (p < 0) ? (-1 - p) : p;
            q = q<targetwidth?q:q%targetwidth;
            xInterpolationIndices.x = q >= targetwidth ? (targetwidth - 1 - q) : q;
            // loop iteration 1
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<targetwidth?q:q%targetwidth;
            xInterpolationIndices.y = q >= targetwidth ? (targetwidth - 1 - q) : q;
            // loop iteration 2
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<targetwidth?q:q%targetwidth;
            xInterpolationIndices.z = q >= targetwidth ? (targetwidth - 1 - q) : q;
            // loop iteration 3
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<targetwidth?q:q%targetwidth;
            xInterpolationIndices.w = q >= targetwidth ? (targetwidth - 1 - q) : q;

            __private int4 yInterpolationIndices;
            p = (coord.y >= 0) ? (((int)trunc(coord.y)) + 2) : (((int)trunc(coord.y)) + 1);
            //loop iteration 0
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.x = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth); // this is the linear absolute index NOT the row
            // loop iteration 1
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.y = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth); // this is the linear absolute index NOT the row
            // loop iteration 2
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.z = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth); // this is the linear absolute index NOT the row
            // loop iteration 3
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.w = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth); // this is the linear absolute index NOT the row

            coord.x -= (coord.x >= dZero) ? (double)((int)trunc(coord.x)) : (double)(((int)trunc(coord.x)) - 1); // get the residual should also be possible with floor
            coord.y -= (coord.y >= dZero) ? (double)((int)trunc(coord.y)) : (double)(((int)trunc(coord.y)) - 1); // get the residual should also be possible with floor

            // Calculate the weights for interpolation
            __private double4 xWeights;
            __private double s = dOne - coord.x;
            xWeights.w = pown(s,3) / dLambda;
            s = coord.x * coord.x;
            xWeights.z = dTwo / dThree - dh * s * (dTwo - coord.x);
            xWeights.x = s * coord.x / dLambda;
            xWeights.y = dOne - xWeights.x - xWeights.z - xWeights.w;
            __private double4 yWeights;
            s = dOne - coord.y;
            yWeights.w = pown(s,3) / dLambda;
            s = coord.y * coord.y;
            yWeights.z = dTwo / dThree - dh * s * (dTwo - coord.y);
            yWeights.x = s * coord.y / dLambda;
            yWeights.y = dOne - yWeights.x - yWeights.z - yWeights.w;

            // unwrapped interpolation loop (sorry 16 iterations) 4x4 interpolation coefficients
            // y loop 0
            s = yWeights.x * (xWeights.x * target[yInterpolationIndices.x + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.x + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.x + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.x + xInterpolationIndices.w]);
            // y loop 1
            s += yWeights.y * (xWeights.x * target[yInterpolationIndices.y + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.y + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.y + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.y + xInterpolationIndices.w]);
            // y loop 2
            s += yWeights.z * (xWeights.x * target[yInterpolationIndices.z + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.z + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.z + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.z + xInterpolationIndices.w]);
            // y loop 3
            s += yWeights.w * (xWeights.x * target[yInterpolationIndices.w + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.w + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.w + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.w + xInterpolationIndices.w]);
            // now s is the value
            __private double diff = source[nIndex] - s;
            diffout[nIndex] = pown(diff,2);
            grad0[nIndex] = diff * xGradient[nIndex];
            grad1[nIndex] = diff * yGradient[nIndex];
            hessian00[nIndex] = pown(xGradient[nIndex],2);
            hessian01[nIndex] = xGradient[nIndex] * yGradient[nIndex];
            hessian11[nIndex] = pown(yGradient[nIndex],2);
        }
        else
        {
            grad0[nIndex] = dZero;
            grad1[nIndex] = dZero;
            hessian00[nIndex] = dZero;
            hessian01[nIndex] = dZero;
            hessian11[nIndex] = dZero;
            diffout[nIndex] = dZero;
            mask[nIndex] = dZero;
        }
    }
}

__kernel void dsumInLocalMemory(__global double *gdata, __local volatile double* ldata, const int size)
{
    /*
    Only a single workgroup must be started but that won't be tested here.
    The number of threads MUST be at least (size + (size % 2))/2!!!!!
    Watch out for the barriers. A barrier must be reached by all threads,
    therefore you can't use a global return statement to get rid of them,
    rather you just define behavior for the threads you want to be active
    and then let all threads hit the barriers.
    */
    __private int nIndex = get_local_id(0);
    __private int nrOfElems = size + (MODULO2(size)); // now it's divisible by two
    __private int divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata[nIndex] + gdata[nIndex + divs]; // linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata[nIndex];
        }
    }
    // Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE); // Synchronize the local memory access
    /*
    Unlike the examples presented by NVidia we don't have the luxury of assuming that n is a power of two
    meaning for example that the first step is 14 which is %2 = 0 but 14/2=7 which is %2 = 1!!!
    This forces us to check for every tree step iteration whether the step is %2 = 0
    */
    // Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata[nIndex] = ldata[nIndex]; // transfer back to global memory
    }
}

__kernel void dsumInLocalMemoryCombined(__global double *gdata0,__global double *gdata1,__global double *gdata2,__global double *gdata3,__global double *gdata4,__global double *gdata5,__global double *gdata6,__global double *gdata7,__global double *gdata8,__global double *gdata9,__global double *gdata10, __local volatile double* ldata, const int size)
{
    /*
    Only a single workgroup must be started but that won't be tested here.
    The number of threads MUST be at least (size + (size % 2))/2!!!!!
    Watch out for the barriers. A barrier must be reached by all threads,
    therefore you can't use a global return statement to get rid of them,
    rather you just define behavior for the threads you want to be active
    and then let all threads hit the barriers.
    */
    __private int nIndex = get_local_id(0);
    __private int nrOfElems = size + (MODULO2(size)); // now it's divisible by two
    __private int divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata0[nIndex] + gdata0[nIndex + divs]; // linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata0[nIndex];
        }
    }
    // Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE); // Synchronize the local memory access
    /*
    Unlike the examples presented by NVidia we don't have the luxury of assuming that n is a power of two
    meaning for example the first step is 14 which is %2 = 0 but 14/2=7 which is %2 = 1!!!
    This forces us to check for every tree step iteration whether the step is %2 = 0
    */
    // Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata0[nIndex] = ldata[nIndex]; // transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size)); // now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata1[nIndex] + gdata1[nIndex + divs]; // linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata1[nIndex];
        }
    }
    // Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE); // Synchronize the local memory access
    // Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata1[nIndex] = ldata[nIndex]; // transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size)); // now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata2[nIndex] + gdata2[nIndex + divs]; // linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata2[nIndex];
        }
    }
    // Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE); // Synchronize the local memory access
    // Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata2[nIndex] = ldata[nIndex]; // transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size)); // now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata3[nIndex] + gdata3[nIndex + divs];// linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata3[nIndex];
        }
    }
    // Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE); // Synchronize the local memory access
    // Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata3[nIndex] = ldata[nIndex]; // transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size)); // now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata4[nIndex] + gdata4[nIndex + divs]; // linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata4[nIndex];
        }
    }
    // Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    // Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata4[nIndex] = ldata[nIndex]; // transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size)); // now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata5[nIndex] + gdata5[nIndex + divs]; // linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata5[nIndex];
        }
    }
    // Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE); // Synchronize the local memory access
    // Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata5[nIndex] = ldata[nIndex]; // transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size)); // now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata6[nIndex] + gdata6[nIndex + divs]; // linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata6[nIndex];
        }
    }
    // Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE); // Synchronize the local memory access
    // Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata6[nIndex] = ldata[nIndex]; // transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size)); // now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata7[nIndex] + gdata7[nIndex + divs]; // linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata7[nIndex];
        }
    }
    // Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE); // Synchronize the local memory access
    // Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata7[nIndex] = ldata[nIndex]; // transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size)); // now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata8[nIndex] + gdata8[nIndex + divs]; // linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata8[nIndex];
        }
    }
    // Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE); // Synchronize the local memory access
    // Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata8[nIndex] = ldata[nIndex]; // transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size)); // now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata9[nIndex] + gdata9[nIndex + divs]; // linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata9[nIndex];
        }
    }
    // Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE); // Synchronize the local memory access
    // Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata9[nIndex] = ldata[nIndex]; // transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size)); // now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata10[nIndex] + gdata10[nIndex + divs]; // linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata10[nIndex];
        }
    }
    // Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE); // Synchronize the local memory access
    // Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata10[nIndex] = ldata[nIndex]; // transfer back to global memory
    }
}


__kernel void dtranslationSumInLocalMemoryCombined(__global double *gdata0,__global double *gdata1,__global double *gdata2,__global double *gdata3,__global double *gdata4,__global double *gdata5,__global double *gdata6, __local volatile double* ldata, const int size)
{
    /*
    Only a single workgroup must be started but that won't be tested here.
    The number of threads MUST be at least (size + (size % 2))/2!!!!!
    Watch out for the barriers. A barrier must be reached by all threads,
    therefore you can't use a global return statement to get rid of them,
    rather you just define behavior for the threads you want to be active
    and then let all threads hit the barriers.
    */
    __private int nIndex = get_local_id(0);
    __private int nrOfElems = size + (MODULO2(size));//now it's divisible by two
    __private int divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata0[nIndex] + gdata0[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata0[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    /*
    Unlike the examples presented by NVidia we don't have the luxury of assuming n being a power of two
    meaning for example the first step is 14 which is %2 = 0 but 14/2=7 which is %2 = 1!!!
    This forces us to check for every tree step iteration whether the step is %2 = 0
    */
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata0[nIndex] = ldata[nIndex];//transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata1[nIndex] + gdata1[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata1[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata1[nIndex] = ldata[nIndex];//transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata2[nIndex] + gdata2[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata2[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata2[nIndex] = ldata[nIndex];//transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata3[nIndex] + gdata3[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata3[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata3[nIndex] = ldata[nIndex];//transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata4[nIndex] + gdata4[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata4[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata4[nIndex] = ldata[nIndex];//transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata5[nIndex] + gdata5[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata5[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata5[nIndex] = ldata[nIndex];//transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata6[nIndex] + gdata6[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata6[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata6[nIndex] = ldata[nIndex];//transfer back to global memory
    }
}

/*
The following code is inspired by the parallel sum reduction according to Brent's
theorem published by NVidia. Unlike their code this is designed to also run without
the requirement of power of two multiples (because some SIMDs seem to use 10 or 20).
This makes the code a lot less optimal, but if someone would like to write a separate
implementation for the different cases (which would be easy) feel free.
*/
__kernel void dparallelGroupedSumReduction(__global const double *gdata, __global double *godata /*only needs to be max block NUMBER (not size) in size*/, unsigned int size, __local volatile double* ldata)
{
    //To save on kernels we're fetching and summing already at the first level
    __private unsigned int nIndex = get_local_id(0);/*ID within a workgroup*/
    __private unsigned int blockSize = get_local_size(0); //this is the stride
    __private unsigned int i = get_group_id(0)*(blockSize*2) + nIndex;
    __private unsigned int gridSize = blockSize*2*get_num_groups(0);
    
    ldata[nIndex] = dZero;//prepare the local memory

    // get_num_groups(0) dynamically tunes the number of elements each thread sums by changing the gridSize
    // get_local_size(0) is then equal to the blocksize (the number of threads running within a block)
    while (i < size)
    {         
        ldata[nIndex] += gdata[i];
        // ensure the read is not out of bounds
        if (i + blockSize < size)
        {
            ldata[nIndex] += gdata[i+blockSize];  
        }
        i += gridSize;
    } 

    barrier(CLK_LOCAL_MEM_FENCE);//only need to synchronize the local memory

    //Now we still need to do a tree reduction in the local memory, but because we can't be sure that the groupsize is a power of two we have to do this the slow way
    i = (blockSize + (MODULO2(blockSize)))/2;
    while(blockSize >= 2)
    {
        if((nIndex < i) && (nIndex + i < blockSize))
        {
            ldata[nIndex] += ldata[nIndex + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        blockSize = i;
        i = (blockSize + (MODULO2(blockSize)))/2;
    }
    
    // write result for this block to global mem 
    if(nIndex == 0)
    {
        godata[get_group_id(0)] = ldata[0];//So in the end we still have max block Nr elements which need to be summed up.
    }
}

__kernel void drigidBodyErrorWithGradAndHessBrent(const __global double *source,
const __global double *target,
const __global double *xGradient,
const __global double *yGradient,
__global double *grad0,
__global double *grad1,
__global double *grad2,
__global double *hessian00,
__global double *hessian01,
__global double *hessian02,
__global double *hessian11,
__global double *hessian12,
__global double *hessian22,
__global double *diffout,
__global double *mask,
__local volatile double *lgrad0,
__local volatile double *lgrad1,
__local volatile double *lgrad2,
__local volatile double *lhessian00,
__local volatile double *lhessian01,
__local volatile double *lhessian02,
__local volatile double *lhessian11,
__local volatile double *lhessian12,
__local volatile double *lhessian22,
__local volatile double *ldiffout,
__local volatile double *lmask,
const int sourcewidth,
const int sourceheight,
const int targetwidth,
const int targetheight,
const double offsetx,
const double offsety,
const double angle,
const int doubleTargetWidth,
const int doubleTargetHeight)
{
    //Brent's theorem optimized version to reduce the following sum reduction to blockSize
    __private unsigned int nIndex = get_local_id(0);/*ID within a workgroup*/
    __private unsigned int blockSize = get_local_size(0); //this is the stride
    __private unsigned int i = get_group_id(0)*(blockSize*2) + nIndex;
    __private unsigned int gridSize = blockSize*2*get_num_groups(0);
    
    //prepare local buffers
    lgrad0[nIndex] = dZero;
    lgrad1[nIndex] = dZero;
    lgrad2[nIndex] = dZero;
    lhessian00[nIndex] = dZero;
    lhessian01[nIndex] = dZero;
    lhessian02[nIndex] = dZero;
    lhessian11[nIndex] = dZero;
    lhessian12[nIndex] = dZero;
    lhessian22[nIndex] = dZero;
    ldiffout[nIndex] = dZero;
    lmask[nIndex] = dZero;

    //These vectors remain the same during the loops
    __private double2 xvec = (double2)(cos(angle),-sin(angle));//Warning: this is not the x vector but it is the vector added in the x direction
    __private double2 yvec = (double2)(-xvec.y,xvec.x);//Warning: this is not the y vector but it is the vector added in the y direction
    while(i < sourcewidth * sourceheight)
    {
        __private int column = i % sourcewidth;
        __private int row = (i - column)/sourcewidth;
        __private double2 coord = (double2)(offsetx, offsety) + ((double)column) * xvec + ((double)row) * yvec;
        
        __private int4 xInterpolationIndices;
        __private int4 yInterpolationIndices;
        __private int4 combinedInterpolationIndices;
        __private double4 xWeights;
        __private double4 yWeights;
        __private int p;
        __private int q;
        __private double s;
        __private double diff;
        __private double Theta;
        
        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            lmask[nIndex] += dOne;
            //Following is the calculation using mirrored boundaries of the x and y indices of the coefficients used for interpolation
            //calculate the x coordinates for interpolation (loop unwrapped) for speed
            p = (coord.x >= 0) ? (((int)trunc(coord.x)) + 2) : (((int)trunc(coord.x)) + 1);
            //loop iteration 0
            /*
            q = (p < 0) ? (-1 - p) : p;
            is a diverging statement, although it incurs more operations the following equivalent will likely be faster
            */
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetWidth?q:q%doubleTargetWidth;
            xInterpolationIndices.x = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
            //loop iteration 1
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetWidth?q:q%doubleTargetWidth;
            xInterpolationIndices.y = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
            //loop iteration 2
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetWidth?q:q%doubleTargetWidth;
            xInterpolationIndices.z = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
            //loop iteration 3
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetWidth?q:q%doubleTargetWidth;
            xInterpolationIndices.w = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;

            p = (coord.y >= 0) ? (((int)trunc(coord.y)) + 2) : (((int)trunc(coord.y)) + 1);
            //loop iteration 0
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetHeight?q:q%doubleTargetHeight;
            yInterpolationIndices.x = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 1
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetHeight?q:q%doubleTargetHeight;
            yInterpolationIndices.y = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 2
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetHeight?q:q%doubleTargetHeight;
            yInterpolationIndices.z = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 3
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetHeight?q:q%doubleTargetHeight;
            yInterpolationIndices.w = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row

            coord.x -= (coord.x >= dZero) ? (double)((int)trunc(coord.x)) : (double)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with floor
            coord.y -= (coord.y >= dZero) ? (double)((int)trunc(coord.y)) : (double)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with floor

            //Calculate the weights for interpolation
            s = dOne - coord.x;
            xWeights.w = pown(s,3) / dLambda;
            s = coord.x * coord.x;
            xWeights.z = dTwo / dThree - dh * s * (dTwo - coord.x);
            xWeights.x = s * coord.x / dLambda;
            xWeights.y = dOne - xWeights.x - xWeights.z - xWeights.w;
            s = dOne - coord.y;
            yWeights.w = pown(s,3) / dLambda;
            s = coord.y * coord.y;
            yWeights.z = dTwo / dThree - dh * s * (dTwo - coord.y);
            yWeights.x = s * coord.y / dLambda;
            yWeights.y = dOne - yWeights.x - yWeights.z - yWeights.w;

            //unwrapped interpolation loop (sorry 16 iterations) 4x4 interpolation coefficients
            //y loop 0
            //broadcast then add
            combinedInterpolationIndices = (int4)(yInterpolationIndices.x) + xInterpolationIndices;
            s = yWeights.x * (xWeights.x * target[combinedInterpolationIndices.x]
                + xWeights.y * target[combinedInterpolationIndices.y]
                + xWeights.z * target[combinedInterpolationIndices.z]
                + xWeights.w * target[combinedInterpolationIndices.w]);
            //y loop 1
            combinedInterpolationIndices = (int4)(yInterpolationIndices.y) + xInterpolationIndices;
            s += yWeights.y * (xWeights.x * target[combinedInterpolationIndices.x]
                + xWeights.y * target[combinedInterpolationIndices.y]
                + xWeights.z * target[combinedInterpolationIndices.z]
                + xWeights.w * target[combinedInterpolationIndices.w]);
            //y loop 2
            combinedInterpolationIndices = (int4)(yInterpolationIndices.z) + xInterpolationIndices;
            s += yWeights.z * (xWeights.x * target[combinedInterpolationIndices.x]
                + xWeights.y * target[combinedInterpolationIndices.y]
                + xWeights.z * target[combinedInterpolationIndices.z]
                + xWeights.w * target[combinedInterpolationIndices.w]);
            //y loop 3
            combinedInterpolationIndices = (int4)(yInterpolationIndices.w) + xInterpolationIndices;
            s += yWeights.w * (xWeights.x * target[combinedInterpolationIndices.x]
                + xWeights.y * target[combinedInterpolationIndices.y]
                + xWeights.z * target[combinedInterpolationIndices.z]
                + xWeights.w * target[combinedInterpolationIndices.w]);
            //now s is the value
            diff = source[i] - s;
            ldiffout[nIndex] += pown(diff,2);
            Theta = yGradient[i] * (double)column - xGradient[i] * (double)row;
            lgrad0[nIndex] += diff * Theta;
            lgrad1[nIndex] += diff * xGradient[i];
            lgrad2[nIndex] += diff * yGradient[i];
            lhessian00[nIndex] += pown(Theta,2);
            lhessian01[nIndex] += Theta * xGradient[i];
            lhessian02[nIndex] += Theta * yGradient[i];
            lhessian11[nIndex] += pown(xGradient[i],2);
            lhessian12[nIndex] += xGradient[i] * yGradient[i];
            lhessian22[nIndex] += pown(yGradient[i],2);
        }
        // ensure the read is not out of bounds
        if(i + blockSize < sourcewidth * sourceheight)
        {
            __private int lIdx = i + blockSize;
            column = lIdx % sourcewidth;
            row = (lIdx - column)/sourcewidth;
            coord = (double2)(offsetx, offsety) + ((double)column) * xvec + ((double)row) * yvec;            
            Msk = (int2)((int)round(coord.x), (int)round(coord.y));
            if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
            {
                lmask[nIndex] += dOne;
                //Following is the calculation using mirrored boundaries of the x and y indices of the coefficients used for interpolation
                //calculate the x coordinates for interpolation (loop unwrapped) for speed
                p = (coord.x >= 0) ? (((int)trunc(coord.x)) + 2) : (((int)trunc(coord.x)) + 1);
                //loop iteration 0
                /*
                q = (p < 0) ? (-1 - p) : p;
                is a diverging statement, although it incurs more operations the following equivalent will likely be faster
                */
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetWidth)
                {
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division, it doesn't yield q
                }
                xInterpolationIndices.x = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
                //loop iteration 1
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetWidth)
                {
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division, it doesn't yield q
                }
                xInterpolationIndices.y = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
                //loop iteration 2
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetWidth)
                {
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division, it doesn't yield q
                }
                xInterpolationIndices.z = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
                //loop iteration 3
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetWidth)
                {
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division, it doesn't yield q
                }
                xInterpolationIndices.w = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;

                p = (coord.y >= 0) ? (((int)trunc(coord.y)) + 2) : (((int)trunc(coord.y)) + 1);
                //loop iteration 0
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetHeight)
                {
                    q -= (doubleTargetHeight) * (q / (doubleTargetHeight)); //WARNING: this is an integer division which may not yield q
                }
                yInterpolationIndices.x = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
                //loop iteration 1
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetHeight)
                {
                    q -= (doubleTargetHeight) * (q / (doubleTargetHeight));//WARNING: this is an integer division which may not yield q
                }
                yInterpolationIndices.y = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
                //loop iteration 2
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetHeight)
                {
                    q -= (doubleTargetHeight) * (q / (doubleTargetHeight));//WARNING: this is an integer division which may not yield q
                }
                yInterpolationIndices.z = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
                //loop iteration 3
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetHeight)
                {
                    q -= (doubleTargetHeight) * (q / (doubleTargetHeight));//WARNING: this is an integer division which may not yield q
                }
                yInterpolationIndices.w = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row

                coord.x -= (coord.x >= dZero) ? (double)((int)trunc(coord.x)) : (double)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with floor
                coord.y -= (coord.y >= dZero) ? (double)((int)trunc(coord.y)) : (double)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with floor

                //Calculate the weights for interpolation
                s = dOne - coord.x;
                xWeights.w = pown(s,3) / dLambda;
                s = coord.x * coord.x;
                xWeights.z = dTwo / dThree - dh * s * (dTwo - coord.x);
                xWeights.x = s * coord.x / dLambda;
                xWeights.y = dOne - xWeights.x - xWeights.z - xWeights.w;
                s = dOne - coord.y;
                yWeights.w = pown(s,3) / dLambda;
                s = coord.y * coord.y;
                yWeights.z = dTwo / dThree - dh * s * (dTwo - coord.y);
                yWeights.x = s * coord.y / dLambda;
                yWeights.y = dOne - yWeights.x - yWeights.z - yWeights.w;

                //unwrapped interpolation loop (sorry 16 iterations) 4x4 interpolation coefficients
                //y loop 0
                //broadcast then add
                combinedInterpolationIndices = (int4)(yInterpolationIndices.x) + xInterpolationIndices;
                s = yWeights.x * (xWeights.x * target[combinedInterpolationIndices.x]
                    + xWeights.y * target[combinedInterpolationIndices.y]
                    + xWeights.z * target[combinedInterpolationIndices.z]
                    + xWeights.w * target[combinedInterpolationIndices.w]);
                //y loop 1
                combinedInterpolationIndices = (int4)(yInterpolationIndices.y) + xInterpolationIndices;
                s += yWeights.y * (xWeights.x * target[combinedInterpolationIndices.x]
                    + xWeights.y * target[combinedInterpolationIndices.y]
                    + xWeights.z * target[combinedInterpolationIndices.z]
                    + xWeights.w * target[combinedInterpolationIndices.w]);
                //y loop 2
                combinedInterpolationIndices = (int4)(yInterpolationIndices.z) + xInterpolationIndices;
                s += yWeights.z * (xWeights.x * target[combinedInterpolationIndices.x]
                    + xWeights.y * target[combinedInterpolationIndices.y]
                    + xWeights.z * target[combinedInterpolationIndices.z]
                    + xWeights.w * target[combinedInterpolationIndices.w]);
                //y loop 3
                combinedInterpolationIndices = (int4)(yInterpolationIndices.w) + xInterpolationIndices;
                s += yWeights.w * (xWeights.x * target[combinedInterpolationIndices.x]
                    + xWeights.y * target[combinedInterpolationIndices.y]
                    + xWeights.z * target[combinedInterpolationIndices.z]
                    + xWeights.w * target[combinedInterpolationIndices.w]);
                //now s is the value
                diff = source[lIdx] - s;
                ldiffout[nIndex] += pown(diff,2);
                Theta = yGradient[lIdx] * (double)column - xGradient[lIdx] * (double)row;
                lgrad0[nIndex] += diff * Theta;
                lgrad1[nIndex] += diff * xGradient[lIdx];
                lgrad2[nIndex] += diff * yGradient[lIdx];
                lhessian00[nIndex] += pown(Theta,2);
                lhessian01[nIndex] += Theta * xGradient[lIdx];
                lhessian02[nIndex] += Theta * yGradient[lIdx];
                lhessian11[nIndex] += pown(xGradient[lIdx],2);
                lhessian12[nIndex] += xGradient[lIdx] * yGradient[lIdx];
                lhessian22[nIndex] += pown(yGradient[lIdx],2);
            }
        }
        i += gridSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);//only need to synchronize the local memory
    //Now we still need to do a tree reduction in the local memory, but because we can't be sure that the groupsize is a power of two we have to do this the slow way
    i = (blockSize + (MODULO2(blockSize)))/2;
    while(blockSize >= 2)
    {
        if((nIndex < i) && (nIndex + i < blockSize))
        {
            lgrad0[nIndex] += lgrad0[nIndex + i];
            lgrad1[nIndex] += lgrad1[nIndex + i];
            lgrad2[nIndex] += lgrad2[nIndex + i];
            lhessian00[nIndex] += lhessian00[nIndex + i];
            lhessian01[nIndex] += lhessian01[nIndex + i];
            lhessian02[nIndex] += lhessian02[nIndex + i];
            lhessian11[nIndex] += lhessian11[nIndex + i];
            lhessian12[nIndex] += lhessian12[nIndex + i];
            lhessian22[nIndex] += lhessian22[nIndex + i];
            ldiffout[nIndex] += ldiffout[nIndex + i];
            lmask[nIndex] += lmask[nIndex + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        blockSize = i;
        i = (blockSize + (MODULO2(blockSize)))/2;
    }
    
    // write result for this block to global mem 
    if(nIndex == 0)
    {
        //So in the end we still have max block Nr elements which need to be summed up.
        i = get_group_id(0);
        grad0[i] = lgrad0[0];
        grad1[i] = lgrad1[0];
        grad2[i] = lgrad2[0];
        hessian00[i] = lhessian00[0];
        hessian01[i] = lhessian01[0];
        hessian02[i] = lhessian02[0];
        hessian11[i] = lhessian11[0];
        hessian12[i] = lhessian12[0];
        hessian22[i] = lhessian22[0];
        diffout[i] = ldiffout[0];
        mask[i] = lmask[0];
    }
}


__kernel void dtranslationErrorWithGradAndHessBrent(const __global double *source,
const __global double *target,
const __global double *xGradient,
const __global double *yGradient,
__global double *grad0,
__global double *grad1,
__global double *hessian00,
__global double *hessian01,
__global double *hessian11,
__global double *diffout,
__global double *mask,
__local volatile double *lgrad0,
__local volatile double *lgrad1,
__local volatile double *lhessian00,
__local volatile double *lhessian01,
__local volatile double *lhessian11,
__local volatile double *ldiffout,
__local volatile double *lmask,
const int sourcewidth,
const int sourceheight,
const int targetwidth,
const int targetheight,
const double offsetx,
const double offsety,
const int doubleTargetWidth,
const int doubleTargetHeight)
{
    //Brent's theorem optimized version to reduce the following sum reduction to blockSize
    __private unsigned int nIndex = get_local_id(0);/*ID within a workgroup*/
    __private unsigned int blockSize = get_local_size(0); //this is the stride
    __private unsigned int i = get_group_id(0)*(blockSize*2) + nIndex;
    __private unsigned int gridSize = blockSize*2*get_num_groups(0);
    
    //prepare local buffers
    lgrad0[nIndex] = dZero;
    lgrad1[nIndex] = dZero;
    lhessian00[nIndex] = dZero;
    lhessian01[nIndex] = dZero;
    lhessian11[nIndex] = dZero;
    ldiffout[nIndex] = dZero;
    lmask[nIndex] = dZero;

    while(i < sourcewidth * sourceheight)
    {
        __private int column = i % sourcewidth;
        __private int row = (i - column)/sourcewidth;
        __private double2 coord = (double2)(offsetx + ((double)column), offsety + ((double)row));
        
        __private int4 xInterpolationIndices;
        __private int4 yInterpolationIndices;
        __private int4 combinedInterpolationIndices;
        __private double4 xWeights;
        __private double4 yWeights;
        __private int p;
        __private int q;
        __private double s;
        __private double diff;
        
        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            lmask[nIndex] += dOne;
            //Following is the calculation using mirrored boundaries of the x and y indices of the coefficients used for interpolation
            //calculate the x coordinates for interpolation (loop unwrapped) for speed
            p = (coord.x >= 0) ? (((int)trunc(coord.x)) + 2) : (((int)trunc(coord.x)) + 1);
            //loop iteration 0
            /*
            q = (p < 0) ? (-1 - p) : p;
            is a diverging statement, although it incurs more operations the following equivalent will likely be faster
            */
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetWidth?q:q%doubleTargetWidth;
            xInterpolationIndices.x = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
            //loop iteration 1
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetWidth?q:q%doubleTargetWidth;
            xInterpolationIndices.y = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
            //loop iteration 2
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetWidth?q:q%doubleTargetWidth;
            xInterpolationIndices.z = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
            //loop iteration 3
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetWidth?q:q%doubleTargetWidth;
            xInterpolationIndices.w = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;

            p = (coord.y >= 0) ? (((int)trunc(coord.y)) + 2) : (((int)trunc(coord.y)) + 1);
            //loop iteration 0
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetHeight?q:q%doubleTargetHeight;
            yInterpolationIndices.x = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 1
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetHeight?q:q%doubleTargetHeight;
            yInterpolationIndices.y = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 2
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetHeight?q:q%doubleTargetHeight;
            yInterpolationIndices.z = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 3
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetHeight?q:q%doubleTargetHeight;
            yInterpolationIndices.w = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row

            coord.x -= (coord.x >= dZero) ? (double)((int)trunc(coord.x)) : (double)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with floor
            coord.y -= (coord.y >= dZero) ? (double)((int)trunc(coord.y)) : (double)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with floor

            //Calculate the weights for interpolation
            s = dOne - coord.x;
            xWeights.w = pown(s,3) / dLambda;
            s = coord.x * coord.x;
            xWeights.z = dTwo / dThree - dh * s * (dTwo - coord.x);
            xWeights.x = s * coord.x / dLambda;
            xWeights.y = dOne - xWeights.x - xWeights.z - xWeights.w;
            s = dOne - coord.y;
            yWeights.w = pown(s,3) / dLambda;
            s = coord.y * coord.y;
            yWeights.z = dTwo / dThree - dh * s * (dTwo - coord.y);
            yWeights.x = s * coord.y / dLambda;
            yWeights.y = dOne - yWeights.x - yWeights.z - yWeights.w;

            //unwrapped interpolation loop (sorry 16 iterations) 4x4 interpolation coefficients
            //y loop 0
            //broadcast then add
            combinedInterpolationIndices = (int4)(yInterpolationIndices.x) + xInterpolationIndices;
            s = yWeights.x * (xWeights.x * target[combinedInterpolationIndices.x]
                + xWeights.y * target[combinedInterpolationIndices.y]
                + xWeights.z * target[combinedInterpolationIndices.z]
                + xWeights.w * target[combinedInterpolationIndices.w]);
            //y loop 1
            combinedInterpolationIndices = (int4)(yInterpolationIndices.y) + xInterpolationIndices;
            s += yWeights.y * (xWeights.x * target[combinedInterpolationIndices.x]
                + xWeights.y * target[combinedInterpolationIndices.y]
                + xWeights.z * target[combinedInterpolationIndices.z]
                + xWeights.w * target[combinedInterpolationIndices.w]);
            //y loop 2
            combinedInterpolationIndices = (int4)(yInterpolationIndices.z) + xInterpolationIndices;
            s += yWeights.z * (xWeights.x * target[combinedInterpolationIndices.x]
                + xWeights.y * target[combinedInterpolationIndices.y]
                + xWeights.z * target[combinedInterpolationIndices.z]
                + xWeights.w * target[combinedInterpolationIndices.w]);
            //y loop 3
            combinedInterpolationIndices = (int4)(yInterpolationIndices.w) + xInterpolationIndices;
            s += yWeights.w * (xWeights.x * target[combinedInterpolationIndices.x]
                + xWeights.y * target[combinedInterpolationIndices.y]
                + xWeights.z * target[combinedInterpolationIndices.z]
                + xWeights.w * target[combinedInterpolationIndices.w]);
            //now s is the value
            diff = source[i] - s;
            ldiffout[nIndex] += pown(diff,2);
            lgrad0[nIndex] += diff * xGradient[i];
            lgrad1[nIndex] += diff * yGradient[i];
            lhessian00[nIndex] += pown(xGradient[i],2);
            lhessian01[nIndex] += xGradient[i] * yGradient[i];
            lhessian11[nIndex] += pown(yGradient[i],2);
        }
        // ensure the read is not out of bounds
        if(i + blockSize < sourcewidth * sourceheight)
        {
            __private int lIdx = i + blockSize;
            column = lIdx % sourcewidth;
            row = (lIdx - column)/sourcewidth;
            coord = (double2)(offsetx + ((double)column), offsety + ((double)row));
            Msk = (int2)((int)round(coord.x), (int)round(coord.y));
            if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
            {
                lmask[nIndex] += dOne;
                //Following is the calculation using mirrored boundaries of the x and y indices of the coefficients used for interpolation
                //calculate the x coordinates for interpolation (loop unwrapped) for speed
                p = (coord.x >= 0) ? (((int)trunc(coord.x)) + 2) : (((int)trunc(coord.x)) + 1);
                //loop iteration 0
                /*
                q = (p < 0) ? (-1 - p) : p;
                is a diverging statement, although it incurs more operations the following equivalent will likely be faster
                */
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetWidth)
                {
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division, it doesn't yield q
                }
                xInterpolationIndices.x = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
                //loop iteration 1
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetWidth)
                {
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division, it doesn't yield q
                }
                xInterpolationIndices.y = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
                //loop iteration 2
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetWidth)
                {
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division, it doesn't yield q
                }
                xInterpolationIndices.z = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
                //loop iteration 3
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetWidth)
                {
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division, it doesn't yield q
                }
                xInterpolationIndices.w = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;

                p = (coord.y >= 0) ? (((int)trunc(coord.y)) + 2) : (((int)trunc(coord.y)) + 1);
                //loop iteration 0
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetHeight)
                {
                    q -= (doubleTargetHeight) * (q / (doubleTargetHeight));//WARNING: this is an integer division which may not yield q
                }
                yInterpolationIndices.x = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
                //loop iteration 1
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetHeight)
                {
                    q -= (doubleTargetHeight) * (q / (doubleTargetHeight));//WARNING: this is an integer division which may not yield q
                }
                yInterpolationIndices.y = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
                //loop iteration 2
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetHeight)
                {
                    q -= (doubleTargetHeight) * (q / (doubleTargetHeight));//WARNING: this is an integer division which may not yield q
                }
                yInterpolationIndices.z = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
                //loop iteration 3
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetHeight)
                {
                    q -= (doubleTargetHeight) * (q / (doubleTargetHeight));//WARNING: this is an integer division which may not yield q
                }
                yInterpolationIndices.w = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row

                coord.x -= (coord.x >= dZero) ? (double)((int)trunc(coord.x)) : (double)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with floor
                coord.y -= (coord.y >= dZero) ? (double)((int)trunc(coord.y)) : (double)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with floor

                //Calculate the weights for interpolation
                s = dOne - coord.x;
                xWeights.w = pown(s,3) / dLambda;
                s = coord.x * coord.x;
                xWeights.z = dTwo / dThree - dh * s * (dTwo - coord.x);
                xWeights.x = s * coord.x / dLambda;
                xWeights.y = dOne - xWeights.x - xWeights.z - xWeights.w;
                s = dOne - coord.y;
                yWeights.w = pown(s,3) / dLambda;
                s = coord.y * coord.y;
                yWeights.z = dTwo / dThree - dh * s * (dTwo - coord.y);
                yWeights.x = s * coord.y / dLambda;
                yWeights.y = dOne - yWeights.x - yWeights.z - yWeights.w;

                //unwrapped interpolation loop (sorry 16 iterations) 4x4 interpolation coefficients
                //y loop 0
                //broadcast then add
                combinedInterpolationIndices = (int4)(yInterpolationIndices.x) + xInterpolationIndices;
                s = yWeights.x * (xWeights.x * target[combinedInterpolationIndices.x]
                    + xWeights.y * target[combinedInterpolationIndices.y]
                    + xWeights.z * target[combinedInterpolationIndices.z]
                    + xWeights.w * target[combinedInterpolationIndices.w]);
                //y loop 1
                combinedInterpolationIndices = (int4)(yInterpolationIndices.y) + xInterpolationIndices;
                s += yWeights.y * (xWeights.x * target[combinedInterpolationIndices.x]
                    + xWeights.y * target[combinedInterpolationIndices.y]
                    + xWeights.z * target[combinedInterpolationIndices.z]
                    + xWeights.w * target[combinedInterpolationIndices.w]);
                //y loop 2
                combinedInterpolationIndices = (int4)(yInterpolationIndices.z) + xInterpolationIndices;
                s += yWeights.z * (xWeights.x * target[combinedInterpolationIndices.x]
                    + xWeights.y * target[combinedInterpolationIndices.y]
                    + xWeights.z * target[combinedInterpolationIndices.z]
                    + xWeights.w * target[combinedInterpolationIndices.w]);
                //y loop 3
                combinedInterpolationIndices = (int4)(yInterpolationIndices.w) + xInterpolationIndices;
                s += yWeights.w * (xWeights.x * target[combinedInterpolationIndices.x]
                    + xWeights.y * target[combinedInterpolationIndices.y]
                    + xWeights.z * target[combinedInterpolationIndices.z]
                    + xWeights.w * target[combinedInterpolationIndices.w]);
                //now s is the value
                diff = source[lIdx] - s;
                ldiffout[nIndex] += pown(diff,2);
                lgrad0[nIndex] += diff * xGradient[lIdx];
                lgrad1[nIndex] += diff * yGradient[lIdx];
                lhessian00[nIndex] += pown(xGradient[lIdx],2);
                lhessian01[nIndex] += xGradient[lIdx] * yGradient[lIdx];
                lhessian11[nIndex] += pown(yGradient[lIdx],2);
            }
        }
        i += gridSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);//only need to synchronize the local memory
    //Now we still need to do a tree reduction in the local memory, but because we can't be sure that the groupsize is a power of two we have to do this the slow way
    i = (blockSize + (MODULO2(blockSize)))/2;
    while(blockSize >= 2)
    {
        if((nIndex < i) && (nIndex + i < blockSize))
        {
            lgrad0[nIndex] += lgrad0[nIndex + i];
            lgrad1[nIndex] += lgrad1[nIndex + i];
            lhessian00[nIndex] += lhessian00[nIndex + i];
            lhessian01[nIndex] += lhessian01[nIndex + i];
            lhessian11[nIndex] += lhessian11[nIndex + i];
            ldiffout[nIndex] += ldiffout[nIndex + i];
            lmask[nIndex] += lmask[nIndex + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        blockSize = i;
        i = (blockSize + (MODULO2(blockSize)))/2;
    }
    
    // write result for this block to global mem 
    if(nIndex == 0)
    {
        //So in the end we still have max block Nr elements which need to be summed up.
        i = get_group_id(0);
        grad0[i] = lgrad0[0];
        grad1[i] = lgrad1[0];
        hessian00[i] = lhessian00[0];
        hessian01[i] = lhessian01[0];
        hessian11[i] = lhessian11[0];
        diffout[i] = ldiffout[0];
        mask[i] = lmask[0];
    }
}

__kernel void dtransformImageWithBsplineInterpolation(const __global double *source ,__global double *target, const int sourcewidth, const int sourceheight, const double offsetx, const double offsety, const double angle)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private double2 xvec = (double2)(cos(angle),-sin(angle));//Warning: this is not the x vector but it is the vector added in the x direction
        __private double2 yvec = (double2)(-xvec.y,xvec.x);//Warning: this is not the y vector but it is the vector added in the y direction
        __private double2 coord = (double2)(offsetx, offsety) + ((double)column) * xvec + ((double)row) * yvec;

        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        if ((Msk.x >= 0) && (Msk.x < sourcewidth) && (Msk.y >= 0) && (Msk.y < sourceheight))
        {
            //Following is the calculation using mirrored boundaries of the x and y indices of the coefficients used for interpolation
            //calculate the x coordinates for interpolation (loop unwrapped) for speed
            __private int4 xInterpolationIndices;
            __private int p = (coord.x >= 0) ? (((int)trunc(coord.x)) + 2) : (((int)trunc(coord.x)) + 1);
            //loop iteration 0
            __private int q = (p < 0) ? (-1 - p) : p;
            if(q >= 2*sourcewidth)
            {
                q -= (2*sourcewidth) * (q / (2*sourcewidth)); //Warning: this is an integer division, it doesn't yield q
            }
            xInterpolationIndices.x = q >= sourcewidth ? (2*sourcewidth - 1 - q) : q;
            //loop iteration 1
            p--;
            q = (p < 0) ? (-1 - p) : p;
            if(q >= 2*sourcewidth)
            {
                q -= (2*sourcewidth) * (q / (2*sourcewidth)); //Warning: this is an integer division, it doesn't yield q
            }
            xInterpolationIndices.y = q >= sourcewidth ? (2*sourcewidth - 1 - q) : q;
            //loop iteration 2
            p--;
            q = (p < 0) ? (-1 - p) : p;
            if(q >= 2*sourcewidth)
            {
                q -= (2*sourcewidth) * (q / (2*sourcewidth)); //Warning: this is an integer division, it doesn't yield q
            }
            xInterpolationIndices.z = q >= sourcewidth ? (2*sourcewidth - 1 - q) : q;
            //loop iteration 3
            p--;
            q = (p < 0) ? (-1 - p) : p;
            if(q >= 2*sourcewidth)
            {
                q -= (2*sourcewidth) * (q / (2*sourcewidth)); //Warning: this is an integer division, it doesn't yield q
            }
            xInterpolationIndices.w = q >= sourcewidth ? (2*sourcewidth - 1 - q) : q;

            __private int4 yInterpolationIndices;
            p = (coord.y >= 0) ? (((int)trunc(coord.y)) + 2) : (((int)trunc(coord.y)) + 1);
            //loop iteration 0
            q = (p < 0) ? (-1 - p) : p;
            if(q >= 2*sourceheight)
            {
                q -= (2*sourceheight) * (q / (2*sourceheight));//WARNING: this is an integer division which may not yield q
            }
            yInterpolationIndices.x = (sourceheight <= q) ? (((2*sourceheight) - 1 - q) * sourcewidth) : (q * sourcewidth);//this is the linear absolute index NOT the row
            //loop iteration 1
            p--;
            q = (p < 0) ? (-1 - p) : p;
            if(q >= 2*sourceheight)
            {
                q -= (2*sourceheight) * (q / (2*sourceheight));//WARNING: this is an integer division which may not yield q
            }
            yInterpolationIndices.y = (sourceheight <= q) ? (((2*sourceheight) - 1 - q) * sourcewidth) : (q * sourcewidth);//this is the linear absolute index NOT the row
            //loop iteration 2
            p--;
            q = (p < 0) ? (-1 - p) : p;
            if(q >= 2*sourceheight)
            {
                q -= (2*sourceheight) * (q / (2*sourceheight));//WARNING: this is an integer division which may not yield q
            }
            yInterpolationIndices.z = (sourceheight <= q) ? (((2*sourceheight) - 1 - q) * sourcewidth) : (q * sourcewidth);//this is the linear absolute index NOT the row
            //loop iteration 3
            p--;
            q = (p < 0) ? (-1 - p) : p;
            if(q >= 2*sourceheight)
            {
                q -= (2*sourceheight) * (q / (2*sourceheight));//WARNING: this is an integer division which may not yield q
            }
            yInterpolationIndices.w = (sourceheight <= q) ? (((2*sourceheight) - 1 - q) * sourcewidth) : (q * sourcewidth);//this is the linear absolute index NOT the row

            coord.x -= (coord.x >= dZero) ? (double)((int)trunc(coord.x)) : (double)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with floor
            coord.y -= (coord.y >= dZero) ? (double)((int)trunc(coord.y)) : (double)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with floor

            //Calculate the weights for interpolation
            __private double4 xWeights;
            __private double s = dOne - coord.x;
            xWeights.w = pown(s,3) / dLambda;
            s = coord.x * coord.x;
            xWeights.z = dTwo / dThree - dh * s * (dTwo - coord.x);
            xWeights.x = s * coord.x / dLambda;
            xWeights.y = dOne - xWeights.x - xWeights.z - xWeights.w;
            __private double4 yWeights;
            s = dOne - coord.y;
            yWeights.w = pown(s,3) / dLambda;
            s = coord.y * coord.y;
            yWeights.z = dTwo / dThree - dh * s * (dTwo - coord.y);
            yWeights.x = s * coord.y / dLambda;
            yWeights.y = dOne - yWeights.x - yWeights.z - yWeights.w;

            //unwrapped interpolation loop (sorry 16 iterations) 4x4 interpolation coefficients
            //y loop 0
            s = yWeights.x * (xWeights.x * source[yInterpolationIndices.x + xInterpolationIndices.x]
                + xWeights.y * source[yInterpolationIndices.x + xInterpolationIndices.y]
                + xWeights.z * source[yInterpolationIndices.x + xInterpolationIndices.z]
                + xWeights.w * source[yInterpolationIndices.x + xInterpolationIndices.w]);
            //y loop 1
            s += yWeights.y * (xWeights.x * source[yInterpolationIndices.y + xInterpolationIndices.x]
                + xWeights.y * source[yInterpolationIndices.y + xInterpolationIndices.y]
                + xWeights.z * source[yInterpolationIndices.y + xInterpolationIndices.z]
                + xWeights.w * source[yInterpolationIndices.y + xInterpolationIndices.w]);
            //y loop 2
            s += yWeights.z * (xWeights.x * source[yInterpolationIndices.z + xInterpolationIndices.x]
                + xWeights.y * source[yInterpolationIndices.z + xInterpolationIndices.y]
                + xWeights.z * source[yInterpolationIndices.z + xInterpolationIndices.z]
                + xWeights.w * source[yInterpolationIndices.z + xInterpolationIndices.w]);
            //y loop 3
            s += yWeights.w * (xWeights.x * source[yInterpolationIndices.w + xInterpolationIndices.x]
                + xWeights.y * source[yInterpolationIndices.w + xInterpolationIndices.y]
                + xWeights.z * source[yInterpolationIndices.w + xInterpolationIndices.z]
                + xWeights.w * source[yInterpolationIndices.w + xInterpolationIndices.w]);
            //now s is the value
            target[nIndex] = s;
        }
        else
        {
            target[nIndex] = dZero;
        } 
    }   
}

__kernel void dtranslationtransformImageWithBsplineInterpolation(const __global double *source ,__global double *target, const int sourcewidth, const int sourceheight, const double offsetx, const double offsety)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private double2 coord = (double2)(offsetx + ((double)column), offsety + ((double)row));

        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        if ((Msk.x >= 0) && (Msk.x < sourcewidth) && (Msk.y >= 0) && (Msk.y < sourceheight))
        {
            //Following is the calculation using mirrored boundaries of the x and y indices of the coefficients used for interpolation
            //calculate the x coordinates for interpolation (loop unwrapped) for speed
            __private int4 xInterpolationIndices;
            __private int p = (coord.x >= 0) ? (((int)trunc(coord.x)) + 2) : (((int)trunc(coord.x)) + 1);
            //loop iteration 0
            __private int q = (p < 0) ? (-1 - p) : p;
            if(q >= 2*sourcewidth)
            {
                q -= (2*sourcewidth) * (q / (2*sourcewidth)); //Warning: this is an integer division, it doesn't yield q
            }
            xInterpolationIndices.x = q >= sourcewidth ? (2*sourcewidth - 1 - q) : q;
            //loop iteration 1
            p--;
            q = (p < 0) ? (-1 - p) : p;
            if(q >= 2*sourcewidth)
            {
                q -= (2*sourcewidth) * (q / (2*sourcewidth)); //Warning: this is an integer division, it doesn't yield q
            }
            xInterpolationIndices.y = q >= sourcewidth ? (2*sourcewidth - 1 - q) : q;
            //loop iteration 2
            p--;
            q = (p < 0) ? (-1 - p) : p;
            if(q >= 2*sourcewidth)
            {
                q -= (2*sourcewidth) * (q / (2*sourcewidth)); //Warning: this is an integer division, it doesn't yield q
            }
            xInterpolationIndices.z = q >= sourcewidth ? (2*sourcewidth - 1 - q) : q;
            //loop iteration 3
            p--;
            q = (p < 0) ? (-1 - p) : p;
            if(q >= 2*sourcewidth)
            {
                q -= (2*sourcewidth) * (q / (2*sourcewidth)); //Warning: this is an integer division, it doesn't yield q
            }
            xInterpolationIndices.w = q >= sourcewidth ? (2*sourcewidth - 1 - q) : q;

            __private int4 yInterpolationIndices;
            p = (coord.y >= 0) ? (((int)trunc(coord.y)) + 2) : (((int)trunc(coord.y)) + 1);
            //loop iteration 0
            q = (p < 0) ? (-1 - p) : p;
            if(q >= 2*sourceheight)
            {
                q -= (2*sourceheight) * (q / (2*sourceheight));//WARNING: this is an integer division which may not yield q
            }
            yInterpolationIndices.x = (sourceheight <= q) ? (((2*sourceheight) - 1 - q) * sourcewidth) : (q * sourcewidth);//this is the linear absolute index NOT the row
            //loop iteration 1
            p--;
            q = (p < 0) ? (-1 - p) : p;
            if(q >= 2*sourceheight)
            {
                q -= (2*sourceheight) * (q / (2*sourceheight));//WARNING: this is an integer division which may not yield q
            }
            yInterpolationIndices.y = (sourceheight <= q) ? (((2*sourceheight) - 1 - q) * sourcewidth) : (q * sourcewidth);//this is the linear absolute index NOT the row
            //loop iteration 2
            p--;
            q = (p < 0) ? (-1 - p) : p;
            if(q >= 2*sourceheight)
            {
                q -= (2*sourceheight) * (q / (2*sourceheight));//WARNING: this is an integer division which may not yield q
            }
            yInterpolationIndices.z = (sourceheight <= q) ? (((2*sourceheight) - 1 - q) * sourcewidth) : (q * sourcewidth);//this is the linear absolute index NOT the row
            //loop iteration 3
            p--;
            q = (p < 0) ? (-1 - p) : p;
            if(q >= 2*sourceheight)
            {
                q -= (2*sourceheight) * (q / (2*sourceheight));//WARNING: this is an integer division which may not yield q
            }
            yInterpolationIndices.w = (sourceheight <= q) ? (((2*sourceheight) - 1 - q) * sourcewidth) : (q * sourcewidth);//this is the linear absolute index NOT the row

            coord.x -= (coord.x >= dZero) ? (double)((int)trunc(coord.x)) : (double)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with floor
            coord.y -= (coord.y >= dZero) ? (double)((int)trunc(coord.y)) : (double)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with floor

            //Calculate the weights for interpolation
            __private double4 xWeights;
            __private double s = dOne - coord.x;
            xWeights.w = pown(s,3) / dLambda;
            s = coord.x * coord.x;
            xWeights.z = dTwo / dThree - dh * s * (dTwo - coord.x);
            xWeights.x = s * coord.x / dLambda;
            xWeights.y = dOne - xWeights.x - xWeights.z - xWeights.w;
            __private double4 yWeights;
            s = dOne - coord.y;
            yWeights.w = pown(s,3) / dLambda;
            s = coord.y * coord.y;
            yWeights.z = dTwo / dThree - dh * s * (dTwo - coord.y);
            yWeights.x = s * coord.y / dLambda;
            yWeights.y = dOne - yWeights.x - yWeights.z - yWeights.w;

            //unwrapped interpolation loop (sorry 16 iterations) 4x4 interpolation coefficients
            //y loop 0
            s = yWeights.x * (xWeights.x * source[yInterpolationIndices.x + xInterpolationIndices.x]
                + xWeights.y * source[yInterpolationIndices.x + xInterpolationIndices.y]
                + xWeights.z * source[yInterpolationIndices.x + xInterpolationIndices.z]
                + xWeights.w * source[yInterpolationIndices.x + xInterpolationIndices.w]);
            //y loop 1
            s += yWeights.y * (xWeights.x * source[yInterpolationIndices.y + xInterpolationIndices.x]
                + xWeights.y * source[yInterpolationIndices.y + xInterpolationIndices.y]
                + xWeights.z * source[yInterpolationIndices.y + xInterpolationIndices.z]
                + xWeights.w * source[yInterpolationIndices.y + xInterpolationIndices.w]);
            //y loop 2
            s += yWeights.z * (xWeights.x * source[yInterpolationIndices.z + xInterpolationIndices.x]
                + xWeights.y * source[yInterpolationIndices.z + xInterpolationIndices.y]
                + xWeights.z * source[yInterpolationIndices.z + xInterpolationIndices.z]
                + xWeights.w * source[yInterpolationIndices.z + xInterpolationIndices.w]);
            //y loop 3
            s += yWeights.w * (xWeights.x * source[yInterpolationIndices.w + xInterpolationIndices.x]
                + xWeights.y * source[yInterpolationIndices.w + xInterpolationIndices.y]
                + xWeights.z * source[yInterpolationIndices.w + xInterpolationIndices.z]
                + xWeights.w * source[yInterpolationIndices.w + xInterpolationIndices.w]);
            //now s is the value
            target[nIndex] = s;
        }
        else
        {
            target[nIndex] = dZero;
        } 
    }   
}
#endif




//always expose the float code
__constant const float fLambda = 6.0f;
__constant const float fPole = -0.26794919243112270647255365849413f;
__constant const float fOne = 1.0f;
__constant const float fTwo = 2.0f;
__constant const float fh0D3 = 0.66666666666666666666666666666666666666666666666666666666666666667f;
__constant const float fh1D3 = 0.16666666666666666666666666666666666666666666666666666666666666667f;
__constant const int Horizon = 12;

__constant const float fZ0 = -0.5352804307964381655424037816816460718339231523426924148812f;
__constant const float fZ1 = -0.122554615192326690515272264359357343605486549427295558490763f;
__constant const float fZ2 = -0.0091486948096082769285930216516478534156925639545994482648003f;
__constant const float fLambda7 = 5040.0f;
__constant const float fh0D7 = 0.4793650793650793650793650793650793650793650793650793650793650793651f;
__constant const float fh1D7 = 0.23630952380952380952380952380952380952380952380952380952380952380952f;
__constant const float fh2D7 = 0.023809523809523809523809523809523809523809523809523809523809523810f;
__constant const float fh3D7 = 0.00019841269841269841269841269841269841269841269841269841269841269841f;

__constant const float frh0 = 0.375f;
__constant const float frh1 = 0.25f;
__constant const float frh2 = 0.0625f;

__constant const float fh = 0.5f;
__constant const float fZero = 0.0f;
__constant const float fThree = 3.0;

__kernel void fTargetedCubicBSplinePrefilter2Dpremulhp(__global float *image, __global float *target, const int size)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the offset
    if(nIndex < size)
    {
        target[nIndex] = image[nIndex] * fLambda;
    }
}

//low precision variant with horizon for float precision (faster) from 10.1093/comjnl/bxq086
__kernel void fCubicBSplinePrefilter2DXlp(__global float *image, const int width, const int height)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the row!!!
    if(nIndex < height)
    {
        __global float *prow = image + (nIndex * width);
        //causal initialization
        __private float zk = fPole;
        __private float Sum = prow[0];
        for(int k = 0; k < Horizon; k++)
        {
            Sum += zk * prow[k];
            zk *= fPole;
        }
        prow[0] = Sum;
        //Causal recursion
        for(int k = 1; k < width; k++)
        {
            prow[k] += fPole *  prow[k-1];
        }
        //anticausal initialization
        prow[width - 1] = (fPole * prow[width - 1] / (fPole - fOne));

        //Anticausal recursion
        for(int k = width - 2; 0 <= k; k--)
        {
            prow[k] = fPole * (prow[k+1] -  prow[k]);
        }
    }
}
//low precision variant with horizon for float precision (faster) from 10.1093/comjnl/bxq086
__kernel void fCubicBSplinePrefilter2DYlp(__global float *image, const int width, const int height)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the column!!!
    if(nIndex < width)
    {
        __global float *pcol = image + nIndex;
    
        //causal initialization
        __private float zk = fPole;
        __private float Sum = pcol[0];
        for(int k = 0; k < Horizon; k++)
        {
            Sum += zk * pcol[k * width];
            zk *= fPole;
        }
        pcol[0] = Sum;

        //Causal recursion
        for(int k = 1; k < height; k++)
        {
            pcol[k * width] += fPole *  pcol[(k-1)*width];
        }

        //anticausal initialization
        pcol[(height - 1)*width] = (fPole * pcol[(height - 1)*width] / (fPole - fOne));

        //Anticausal recursion
        for(int k = height - 2; 0 <= k; k--)
        {
            pcol[k*width] = (fPole * (pcol[(k+1)*width] -  pcol[k*width]));
        }
    }    
}

__kernel void fBasicToCardinal2DXhp(__global float *image,__global float *target, const int width, const int height)
{
    /*
    *   This is a FIR filter so the calculation for each !TARGET! pixel can be run simultaneously
    *   Thus one also has to think the other way around not source -> target but
    *   rather target <- source
    */
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear pixel address
    if(nIndex < width*height)
    {
        //calculate the current column
        __private int col = nIndex % width;
        __private int row = (nIndex - col)/width;
        //symmetricFirMirrorOffBounds1D
        if(col > 0 && col < (width-1))
        {
            //most common case
            target[nIndex] = fh0D3 * image[nIndex] + fh1D3 * (image[nIndex - 1] + image[nIndex + 1]);
        }
        else if(col == (width-1))
        {
            /* nIndex is already row*width+width-1 so we need not waste calculation power to get this number again */
            target[nIndex] = fh0D3 * image[nIndex] + fh1D3 * (image[nIndex-1] + image[nIndex]);
        }
        else
        {
            //col == 0
            /* nIndex is already row*width+0 so we need not waste calculation power to get this number again */
            target[nIndex] = fh0D3 * image[nIndex] + fh1D3 * (image[nIndex] + image[nIndex+1]);
        }
    }
}
__kernel void fBasicToCardinal2DYhp(__global float *image ,__global float *target, const int width, const int height)
{
    /*
    *   This is a FIR filter so the calculation for each !TARGET! pixel can be run simultaneously
    *   Thus one also has to think the other way around not source -> target but
    *   rather target <- source
    */
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear pixel address
    if(nIndex < width*height)
    {
        //calculate the current column
        __private int col = nIndex % width;
        __private int row = (nIndex - col)/width;
        //symmetricFirMirrorOffBounds1D
        if(row > 0 && row < (height-1))
        {
            //most common case
            target[nIndex] = fh0D3 * image[nIndex] + fh1D3 * (image[(row-1)*width+col] + image[(row+1)*width+col]);
        }
        else if(row == (height-1))
        {
            /* nIndex is already (height-1)*width+col so we need not waste calculation power to get this number again */
            target[nIndex] = fh0D3 * image[nIndex] + fh1D3 * (image[(row-1)*width+col] + image[nIndex]);
        }
        else
        {
            //row == 0
            /* nIndex is already col so we need not waste calculation power to get this number again */
            target[nIndex] = fh0D3 * image[nIndex] + fh1D3 * (image[nIndex] + image[width+col]);
        }
    }
}

__kernel void fCubicBSplinePrefilter2DDeg7premulhp(__global float *image, const int size)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the offset
    if(nIndex < size)
    {
        image[nIndex] *= fLambda7;
    }
}

//low precision variant with horizon for float precision (faster) based on 10.1093/comjnl/bxq086
__kernel void fCubicBSplinePrefilter2DXDeg7lp(__global float *image, const int width, const int height)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the row!!!
    if(nIndex < height)
    {
        __global float *prow = image + (nIndex * width);
        //For beta 7th order this has to be done 3 times
        //Iteration 1

        //causal initialization
        __private float zk = fZ0;
        __private float Sum = prow[0];
        for(int k = 0; k < Horizon; k++)
        {
            Sum += zk * prow[k];
            zk *= fZ0;
        }
        prow[0] = Sum;

    
        //Causal recursion
        for(int k = 1; k < width; k++)
        {
            prow[k] += fZ0 *  prow[k-1];
        }
        //anticausal initialization
        prow[width - 1] = fZ0 * prow[width - 1] / (fZ0 - fOne);

        //Anticausal recursion
        for(int k = width - 2; 0 <= k; k--)
        {
            prow[k] = fZ0 * (prow[k+1] -  prow[k]);
        }
        //Iteration 2
        //causal initialization
        zk = fZ1;
        Sum = prow[0];
        for(int k = 0; k < Horizon; k++)
        {
            Sum += zk * prow[k];
            zk *= fZ1;
        }
        prow[0] = Sum;

        //Causal recursion
        for(int k = 1; k < width; k++)
        {
            prow[k] += fZ1 *  prow[k-1];
        }
        //anticausal initialization
        prow[width - 1] = (fZ1 * prow[width - 1] / (fZ1 - fOne));

        //Anticausal recursion
        for(int k = width - 2; 0 <= k; k--)
        {
            prow[k] = (fZ1 * ( prow[k+1] -  prow[k]));
        }
        //Iteration 3
        //causal initialization
        zk = fZ2;
        Sum = prow[0];
        for(int k = 0; k < Horizon; k++)
        {
            Sum += zk * prow[k];
            zk *= fZ2;
        }
        prow[0] = Sum;
    
        //Causal recursion
        for(int k = 1; k < width; k++)
        {
            prow[k] += fZ2 * prow[k-1];
        }
        //anticausal initialization
        prow[width - 1] = (fZ2 *  prow[width - 1] / (fZ2 - fOne));

        //Anticausal recursion
        for(int k = width - 2; 0 <= k; k--)
        {
            prow[k] = fZ2 * ( prow[k+1] -  prow[k]);
        }
    }
}

//low precision variant with horizon for float precision (faster) based on 10.1093/comjnl/bxq086
__kernel void fCubicBSplinePrefilter2DYDeg7lp(__global float *image, const int width, const int height)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the column!!!
    if(nIndex < width)
    {
        __global float *prow = image + nIndex;
        //For beta 7th order this has to be done 3 times
        //Iteration 1
        //causal initialization
        __private float zk = fZ0;
        __private float Sum = prow[0];
        for(int k = 0; k < Horizon; k++)
        {
            Sum += zk * prow[k * width];
            zk *= fZ0;
        }
        prow[0] = Sum;

        //Causal recursion
        for(int k = 1; k < height; k++)
        {
            prow[k*width] += fZ0 *  prow[(k-1)*width];
        }
        //anticausal initialization
        prow[(height - 1)*width] = fZ0 *  prow[(height - 1)*width] / (fZ0 - fOne);

        //Anticausal recursion
        for(int k = height - 2; 0 <= k; k--)
        {
            prow[k*width] = fZ0 * ( prow[(k+1)*width] -  prow[k*width]);
        }
        //Iteration 2
        //causal initialization
        zk = fZ1;
        Sum = prow[0];
        for(int k = 0; k < Horizon; k++)
        {
            Sum += zk * prow[k * width];
            zk *= fZ1;
        }
        prow[0] = Sum;
    
        //Causal recursion
        for(int k = 1; k < height; k++)
        {
            prow[k*width] += fZ1 *  prow[(k-1)*width];
        }
        //anticausal initialization
        prow[(height - 1)*width] = fZ1 *  prow[(height - 1)*width] / (fZ1 - fOne);

        //Anticausal recursion
        for(int k = height - 2; 0 <= k; k--)
        {
            prow[k*width] = fZ1 * ( prow[(k+1)*width] -  prow[k*width]);
        }
        //Iteration 3
        //causal initialization
        zk = fZ2;
        Sum = prow[0];
        for(int k = 0; k < Horizon; k++)
        {
            Sum += zk * prow[k * width];
            zk *= fZ2;
        }
        prow[0] = Sum;
    
        //Causal recursion
        for(int k = 1; k < height; k++)
        {
            prow[k*width] += fZ2 *  prow[(k-1)*width];
        }
        //anticausal initialization
        prow[(height - 1)*width] = (fZ2 *  prow[(height - 1)*width] / (fZ2 - fOne));

        //Anticausal recursion
        for(int k = height - 2; 0 <= k; k--)
        {
            prow[k*width] = fZ2 * ( prow[(k+1)*width] -  prow[k*width]);
        }
    }
}

__kernel void fBasicToCardinal2DXhpDeg7(__global float *image,__global float *target, const int width, const int height)
{
    /*
    *   This is a FIR filter so the calculation for each !TARGET! pixel can be run simultaneously
    *   Thus one also has to think the other way around not source -> target but
    *   rather target <- source
    */
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear pixel address
    if(nIndex < width*height)
    {
        //calculate the current column
        __private int col = nIndex % width;
        __private int row = (nIndex - col)/width;
        //symmetricFirMirrorOffBounds1D
        //width >= 6 is guaranteed
        if(col > 2 && col < (width-3))
        {
            //most common case
            target[nIndex] = fh0D7 * image[nIndex] + fh1D7 * (image[nIndex-1] + image[nIndex+1]) + fh2D7 * (image[nIndex-2] + image[nIndex+2]) + fh3D7 * (image[nIndex-3] + image[nIndex+3]);
        }
        else if(col == 1)
        {
            target[nIndex] = fh0D7 * image[nIndex] + fh1D7 * (image[nIndex-1] + image[nIndex+1]) + fh2D7 * (image[nIndex-1] + image[nIndex+2]) + fh3D7 * (image[nIndex] + image[nIndex+3]);
        }
        else if(col == 2)
        {
            target[nIndex] = fh0D7 * image[nIndex] + fh1D7 * (image[nIndex-1] + image[nIndex+1]) + fh2D7 * (image[nIndex-2] + image[nIndex+2]) + fh3D7 * (image[nIndex-2] + image[nIndex+3]);
        }
        else if(col == (width-3))
        {
            /* nIndex is already row*width+width-3 so we need not waste calculation power to get this number again */
            target[nIndex] = fh0D7 * image[nIndex] + fh1D7 * (image[nIndex-1] + image[nIndex+1]) + fh2D7 * (image[nIndex-2] + image[nIndex+2]) + fh3D7 * (image[nIndex-3] + image[nIndex+2]);
        }
        else if(col == (width-2))
        {
            /* nIndex is already row*width+width-2 so we need not waste calculation power to get this number again */
            target[nIndex] = fh0D7 * image[nIndex] + fh1D7 * (image[nIndex-1] + image[nIndex+1]) + fh2D7 * (image[nIndex-2] + image[nIndex+1]) + fh3D7 * (image[nIndex-3] + image[nIndex]);
        }
        else if(col == (width-1))
        {
            /* nIndex is already row*width+width-1 so we need not waste calculation power to get this number again */
            target[nIndex] = fh0D7 * image[nIndex] + fh1D7 * (image[nIndex-1] + image[nIndex]) + fh2D7 * (image[nIndex-2] + image[nIndex-1]) + fh3D7 * (image[nIndex-3] + image[nIndex-2]);
        }
        else
        {
            //col == 0
            /* nIndex is already row*width+0 so we need not waste calculation power to get this number again */
            target[nIndex] = fh0D7 * image[nIndex] + fh1D7 * (image[nIndex] + image[nIndex+1]) + fh2D7 * (image[nIndex+1] + image[nIndex+2])+ fh3D7 * (image[nIndex+2] + image[nIndex+3]);
        }
    }
}

__kernel void fBasicToCardinal2DYhpDeg7(__global float *image,__global float *target, const int width, const int height)
{
    /*
    *   This is a FIR filter so the calculation for each !TARGET! pixel can be run simultaneously
    *   Thus one also has to think the other way around not source -> target but
    *   rather target <- source
    */
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear pixel address
    if(nIndex < width*height)
    {
        //calculate the current column
        __private int col = nIndex % width;
        __private int row = (nIndex - col)/width;
        //symmetricFirMirrorOffBounds1D
        //height >= 6 is guaranteed
        if(row > 2 && row < (height-3))
        {
            //most common case
            target[nIndex] = fh0D7 * image[nIndex] + fh1D7 * (image[nIndex-width] + image[nIndex+width]) + fh2D7 * (image[nIndex-2*width] + image[nIndex+2*width]) + fh3D7 * (image[nIndex-3*width] + image[nIndex+3*width]);
        }
        else if(row == 1)
        {
            target[nIndex] = fh0D7 * image[nIndex] + fh1D7 * (image[col] + image[2*width+col]) + fh2D7 * (image[col] + image[3*width+col]) + fh3D7 * (image[nIndex] + image[4*width+col]);
        }
        else if(row == 2)
        {
            target[nIndex] = fh0D7 * image[nIndex] + fh1D7 * (image[width+col] + image[3*width+col]) + fh2D7 * (image[col] + image[4*width+col]) + fh3D7 * (image[col] + image[5*width+col]);
        }
        else if(row == (height-3))
        {
            /* nIndex is already row*width+width-3 so we need not waste calculation power to get this number again */
            target[nIndex] = fh0D7 * image[nIndex] + fh1D7 * (image[nIndex-width] + image[nIndex+width]) + fh2D7 * (image[nIndex-2*width] + image[nIndex+2*width]) + fh3D7 * (image[nIndex-3*width] + image[nIndex+2*width]);
        }
        else if(row == (height-2))
        {
            /* nIndex is already row*width+width-2 so we need not waste calculation power to get this number again */
            target[nIndex] = fh0D7 * image[nIndex] + fh1D7 * (image[nIndex-width] + image[nIndex+width]) + fh2D7 * (image[nIndex-2*width] + image[nIndex+2*width]) + fh3D7 * (image[nIndex-3*width] + image[nIndex]);
        }
        else if(row == (height-1))
        {
            /* nIndex is already row*width+width-1 so we need not waste calculation power to get this number again */
            target[nIndex] = fh0D7 * image[nIndex] + fh1D7 * (image[nIndex-width] + image[nIndex]) + fh2D7 * (image[nIndex-2*width] + image[nIndex-width]) + fh3D7 * (image[nIndex-3*width] + image[nIndex-3*width]);
        }
        else
        {
            //row == 0
            /* nIndex is already row*width+0 so we need not waste calculation power to get this number again */
            target[nIndex] = fh0D7 * image[nIndex] + fh1D7 * (image[nIndex] + image[width+col]) + fh2D7 * (image[width+col] + image[2*width+col])+ fh3D7 * (image[2*width+col] + image[3*width+col]);
        }
    }
}

__kernel void freduceDual1DX(__global float *image,__global float *target, const int width, const int height, const int halfwidth)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear pixel address in the !TARGET! image
    __private int col = nIndex % halfwidth;
    __private int row = (nIndex - col)/halfwidth;
    //reduceDual1D
    if(nIndex < halfwidth*height)
    {
        /*
        *   Warning be aware that the coordinate systems of the target and the source image are different now
        *   because the width of the target is only roughly half of that of the source image. The row
        *   is the same though so calculate everything from the corresponding row offset not nIndex
        */
        //halfwidth >= 2 is guaranteed
        if(col > 0 && col < (halfwidth - 1))
        {
            //most common case
            target[nIndex] = frh0 * image[row*width + col*2] + frh1 * (image[row*width + col*2 - 1] + image[row*width + col*2 + 1]) + frh2 * (image[row*width + col*2 - 2] + image[row*width + col*2 + 2]);
        }
        else if(col == halfwidth - 1)
        {
            if(width == (2 * halfwidth))//Yes this can be different if width % 2 != 0
            {
                target[nIndex] = frh0 * image[row*width+width-2] + frh1 * (image[row*width+width-3] + image[row*width+width-1]) + frh2 * (image[row*width+width-4] + image[row*width+width-1]);
            }
            else
            {
                target[nIndex] = frh0 * image[row*width+width-3] + frh1 * (image[row*width+width-4] + image[row*width+width-2]) + frh2 * (image[row*width+width-5] + image[row*width+width-1]);
            }
        }
        else
        {
            //col == 0
            target[nIndex] =  frh0 * image[row*width] + frh1 * (image[row*width] + image[row*width+1]) + frh2 * (image[row*width+1] + image[row*width+2]);
        }
    }
}

__kernel void freduceDual1DY(__global float *image,__global float *target, const int halfwidth, const int height, const int halfheight)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear pixel address in the !TARGET! image
    __private int col = nIndex % halfwidth;
    __private int row = (nIndex - col)/halfwidth;
    //reduceDual1D
    if(nIndex < halfwidth*halfheight)
    {
        /*
        *   Warning be aware that the coordinate systems of the target and the source image are different now
        *   because the width and height of the target are only roughly half of that of the source image. The col
        *   is the same for the Y version
        */
        //halfheight >= 2 is guaranteed
        if(row > 0 && row < (halfheight - 1))
        {
            //most common case
            target[nIndex] = frh0 * image[2*row*halfwidth+col] + frh1 * (image[(2*row - 1)*halfwidth+col] + image[(2*row + 1)*halfwidth+col]) + frh2 * (image[(2*row - 2)*halfwidth+col] + image[(2*row + 2)*halfwidth+col]);                
        }
        else if(row == halfheight - 1)
        {
            if(height == (2 * halfheight))//Yes this can be different if height % 2 != 0
            {
                target[nIndex] = frh0 * image[(height - 2)*halfwidth+col] + frh1 * (image[(height - 3)*halfwidth+col] + image[(height - 1)*halfwidth+col]) + frh2 * (image[(height - 4)*halfwidth+col] + image[(height - 1)*halfwidth+col]);
            }
            else
            {
                target[nIndex] = frh0 * image[(height - 3)*halfwidth+col] + frh1 * (image[(height - 4)*halfwidth+col] + image[(height - 2)*halfwidth+col]) + frh2 * (image[(height - 5)*halfwidth+col] + image[(height - 1)*halfwidth+col]);
            }
        }
        else
        {
            //row == 0
            target[nIndex] =  frh0 * image[col] + frh1 * (image[col] + image[halfwidth+col]) + frh2 * (image[halfwidth+col] + image[2*halfwidth+col]);
        }
    }
}

__kernel void fantiSymmetricFirMirrorOffBounds1DX(__global float *image ,__global float *target , const int width, const int height)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !TARGET! pixel
    if(nIndex < height * width)
    {
        __private int column = nIndex % width;
        __private int row = (nIndex - column)/width;
        if(column > 0 && column < (width - 1))
        {
            //most common case
            target[nIndex] = fh * (image[nIndex + 1] - image[nIndex - 1]);
        }
        else if(column == width - 1)
        {
            target[nIndex] = fh * (image[nIndex] - image[nIndex - 1]);
        }
        else
        {
            //column == 0
            target[nIndex] = fh * (image[nIndex + 1] - image[nIndex]);
        }
    }
}

__kernel void fantiSymmetricFirMirrorOffBounds1DY(__global float *image ,__global float *target , const int width, const int height)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !TARGET! pixel
    if(nIndex < height * width)
    {
        __private int column = nIndex % width;
        __private int row = (nIndex - column)/width;
        if(row > 0 && row < (height - 1))
        {
            //most common case
            target[nIndex] = fh * (image[(row+1)*width+column] - image[(row-1)*width+column]);
        }
        else if(row == height - 1)
        {
            target[nIndex] = fh * (image[nIndex] - image[(row-1)*width+column]);
        }
        else
        {
            //row == 0
            target[nIndex] = fh * (image[width+column] - image[nIndex]);
        }
    }
}

__kernel void frigidBodyError(const __global float *source ,const __global float *target, __global float *diffout, __global float *mask, const int sourcewidth, const int sourceheight, const int targetwidth, const int targetheight, const float offsetx, const float offsety, const float angle)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private float2 xvec = (float2)(cos(angle),-sin(angle));//Warning: this is not the x vector but it is the vector added in the x direction
        __private float2 yvec = (float2)(-xvec.y,xvec.x);//Warning: this is not the y vector but it is the vector added in the y direction
        __private float2 coord = (float2)(offsetx, offsety) + ((float)column) * xvec + ((float)row) * yvec;

        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        __private int doubletargetwidth = 2*targetwidth;
        __private int doubletargetheight = 2*targetheight;
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            mask[nIndex] = fOne;
            //Following is the calculation using mirrored boundaries of the x and y indices of the coefficients used for interpolation
            //calculate the x coordinates for interpolation (loop unwrapped) for speed
            __private int4 xInterpolationIndices;
            __private int p = (coord.x >= 0) ? (((int)trunc(coord.x)) + 2) : (((int)trunc(coord.x)) + 1);
            //loop iteration 0
            __private int q = (p < 0) ? (-1 - p) : p;
            //kick out divergence (but calculating the modulo may actually be slower than divergence so maybe use a slightly diverging statement)
            q = q<doubletargetwidth?q:q%doubletargetwidth;
            //q %= doubletargetwidth; //will allways give the right answer
            /*
            if(q >= doubletargetwidth)
            {
                //q -= (2*targetwidth) * (q / (2*targetwidth)); //Warning: this is an integer division, it doesn't yield q (in fact it is a simple modulo operation)
                
            }
            */
            xInterpolationIndices.x = q >= targetwidth ? (doubletargetwidth - 1 - q) : q;
            //loop iteration 1
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetwidth?q:q%doubletargetwidth;
            xInterpolationIndices.y = q >= targetwidth ? (doubletargetwidth - 1 - q) : q;
            //loop iteration 2
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetwidth?q:q%doubletargetwidth;
            xInterpolationIndices.z = q >= targetwidth ? (doubletargetwidth - 1 - q) : q;
            //loop iteration 3
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetwidth?q:q%doubletargetwidth;
            xInterpolationIndices.w = q >= targetwidth ? (doubletargetwidth - 1 - q) : q;

            __private int4 yInterpolationIndices;
            p = (coord.y >= 0) ? (((int)trunc(coord.y)) + 2) : (((int)trunc(coord.y)) + 1);
            //loop iteration 0
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.x = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 1
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.y = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 2
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.z = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 3
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.w = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row

            coord.x -= (coord.x >= fZero) ? (float)((int)trunc(coord.x)) : (float)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with trunc
            coord.y -= (coord.y >= fZero) ? (float)((int)trunc(coord.y)) : (float)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with trunc

            //Calculate the weights for interpolation
            __private float4 xWeights;
            __private float s = fOne - coord.x;
            xWeights.w = pown(s,3) / fLambda;
            s = coord.x * coord.x;
            xWeights.z = fTwo / fThree - fh * s * (fTwo - coord.x);
            xWeights.x = s * coord.x / fLambda;
            xWeights.y = fOne - xWeights.x - xWeights.z - xWeights.w;
            __private float4 yWeights;
            s = fOne - coord.y;
            yWeights.w = pown(s,3) / fLambda;
            s = coord.y * coord.y;
            yWeights.z = fTwo / fThree - fh * s * (fTwo - coord.y);
            yWeights.x = s * coord.y / fLambda;
            yWeights.y = fOne - yWeights.x - yWeights.z - yWeights.w;

            //unwrapped interpolation loop (sorry 16 iterations) 4x4 interpolation coefficients
            //y loop 0
            s = yWeights.x * (xWeights.x * target[yInterpolationIndices.x + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.x + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.x + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.x + xInterpolationIndices.w]);
            //y loop 1
            s += yWeights.y * (xWeights.x * target[yInterpolationIndices.y + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.y + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.y + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.y + xInterpolationIndices.w]);
            //y loop 2
            s += yWeights.z * (xWeights.x * target[yInterpolationIndices.z + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.z + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.z + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.z + xInterpolationIndices.w]);
            //y loop 3
            s += yWeights.w * (xWeights.x * target[yInterpolationIndices.w + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.w + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.w + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.w + xInterpolationIndices.w]);
            //now s is the value
            diffout[nIndex] = pown(source[nIndex] - s,2);
        }
        else
        {
            diffout[nIndex] = fZero;
            mask[nIndex] = fZero;
        } 
    }   
}

__kernel void ftranslationError(const __global float *source ,const __global float *target, __global float *diffout, __global float *mask, const int sourcewidth, const int sourceheight, const int targetwidth, const int targetheight, const float offsetx, const float offsety)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private float2 coord = (float2)(offsetx + ((float)column), offsety + ((float)row));

        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        __private int doubletargetwidth = 2*targetwidth;
        __private int doubletargetheight = 2*targetheight;
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            mask[nIndex] = fOne;
            //Following is the calculation using mirrored boundaries of the x and y indices of the coefficients used for interpolation
            //calculate the x coordinates for interpolation (loop unwrapped) for speed
            __private int4 xInterpolationIndices;
            __private int p = (coord.x >= 0) ? (((int)trunc(coord.x)) + 2) : (((int)trunc(coord.x)) + 1);
            //loop iteration 0
            __private int q = (p < 0) ? (-1 - p) : p;
            //kick out divergence (but calculating the modulo may actually be slower than divergence so maybe use a slightly diverging statement)
            q = q<doubletargetwidth?q:q%doubletargetwidth;
            //q %= doubletargetwidth; //will allways give the right answer
            /*
            if(q >= doubletargetwidth)
            {
                //q -= (2*targetwidth) * (q / (2*targetwidth)); //Warning: this is an integer division, it doesn't yield q (in fact it is a simple modulo operation)
                
            }
            */
            xInterpolationIndices.x = q >= targetwidth ? (doubletargetwidth - 1 - q) : q;
            //loop iteration 1
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetwidth?q:q%doubletargetwidth;
            xInterpolationIndices.y = q >= targetwidth ? (doubletargetwidth - 1 - q) : q;
            //loop iteration 2
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetwidth?q:q%doubletargetwidth;
            xInterpolationIndices.z = q >= targetwidth ? (doubletargetwidth - 1 - q) : q;
            //loop iteration 3
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetwidth?q:q%doubletargetwidth;
            xInterpolationIndices.w = q >= targetwidth ? (doubletargetwidth - 1 - q) : q;

            __private int4 yInterpolationIndices;
            p = (coord.y >= 0) ? (((int)trunc(coord.y)) + 2) : (((int)trunc(coord.y)) + 1);
            //loop iteration 0
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.x = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 1
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.y = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 2
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.z = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 3
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.w = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row

            coord.x -= (coord.x >= fZero) ? (float)((int)trunc(coord.x)) : (float)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with trunc
            coord.y -= (coord.y >= fZero) ? (float)((int)trunc(coord.y)) : (float)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with trunc

            //Calculate the weights for interpolation
            __private float4 xWeights;
            __private float s = fOne - coord.x;
            xWeights.w = pown(s,3) / fLambda;
            s = coord.x * coord.x;
            xWeights.z = fTwo / fThree - fh * s * (fTwo - coord.x);
            xWeights.x = s * coord.x / fLambda;
            xWeights.y = fOne - xWeights.x - xWeights.z - xWeights.w;
            __private float4 yWeights;
            s = fOne - coord.y;
            yWeights.w = pown(s,3) / fLambda;
            s = coord.y * coord.y;
            yWeights.z = fTwo / fThree - fh * s * (fTwo - coord.y);
            yWeights.x = s * coord.y / fLambda;
            yWeights.y = fOne - yWeights.x - yWeights.z - yWeights.w;

            //unwrapped interpolation loop (sorry 16 iterations) 4x4 interpolation coefficients
            //y loop 0
            s = yWeights.x * (xWeights.x * target[yInterpolationIndices.x + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.x + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.x + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.x + xInterpolationIndices.w]);
            //y loop 1
            s += yWeights.y * (xWeights.x * target[yInterpolationIndices.y + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.y + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.y + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.y + xInterpolationIndices.w]);
            //y loop 2
            s += yWeights.z * (xWeights.x * target[yInterpolationIndices.z + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.z + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.z + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.z + xInterpolationIndices.w]);
            //y loop 3
            s += yWeights.w * (xWeights.x * target[yInterpolationIndices.w + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.w + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.w + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.w + xInterpolationIndices.w]);
            //now s is the value
            diffout[nIndex] = pown(source[nIndex] - s,2);
        }
        else
        {
            diffout[nIndex] = fZero;
            mask[nIndex] = fZero;
        } 
    }   
}

__kernel void frigidBodyErrorWithGradAndHess(const __global float *source ,const __global float *target,const __global float *xGradient,const __global float *yGradient,__global float *grad0,__global float *grad1,__global float *grad2,__global float *hessian00,__global float *hessian01,__global float *hessian02,__global float *hessian11,__global float *hessian12,__global float *hessian22, __global float *diffout, __global float *mask, const int sourcewidth, const int sourceheight, const int targetwidth, const int targetheight, const float offsetx, const float offsety, const float angle)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private float2 xvec = (float2)(cos(angle),-sin(angle));//Warning: this is not the x vector but it is the vector added in the x direction
        __private float2 yvec = (float2)(-xvec.y,xvec.x);//Warning: this is not the y vector but it is the vector added in the y direction
        __private float2 coord = (float2)(offsetx, offsety) + ((float)column) * xvec + ((float)row) * yvec;

        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        __private int doubletargetwidth = 2*targetwidth;
        __private int doubletargetheight = 2*targetheight;
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            mask[nIndex] = fOne;
            //Following is the calculation using mirrored boundaries of the x and y indices of the coefficients used for interpolation
            //calculate the x coordinates for interpolation (loop unwrapped) for speed
            __private int4 xInterpolationIndices;
            __private int p = (coord.x >= 0) ? (((int)trunc(coord.x)) + 2) : (((int)trunc(coord.x)) + 1);
            //loop iteration 0
            __private int q = (p < 0) ? (-1 - p) : p;
            q = q<targetwidth?q:q%targetwidth;
            xInterpolationIndices.x = q >= targetwidth ? (targetwidth - 1 - q) : q;
            //loop iteration 1
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<targetwidth?q:q%targetwidth;
            xInterpolationIndices.y = q >= targetwidth ? (targetwidth - 1 - q) : q;
            //loop iteration 2
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<targetwidth?q:q%targetwidth;
            xInterpolationIndices.z = q >= targetwidth ? (targetwidth - 1 - q) : q;
            //loop iteration 3
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<targetwidth?q:q%targetwidth;
            xInterpolationIndices.w = q >= targetwidth ? (targetwidth - 1 - q) : q;

            __private int4 yInterpolationIndices;
            p = (coord.y >= 0) ? (((int)trunc(coord.y)) + 2) : (((int)trunc(coord.y)) + 1);
            //loop iteration 0
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.x = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 1
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.y = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 2
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.z = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 3
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.w = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row

            coord.x -= (coord.x >= fZero) ? (float)((int)trunc(coord.x)) : (float)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with floor
            coord.y -= (coord.y >= fZero) ? (float)((int)trunc(coord.y)) : (float)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with floor

            //Calculate the weights for interpolation
            __private float4 xWeights;
            __private float s = fOne - coord.x;
            xWeights.w = pown(s,3) / fLambda;
            s = coord.x * coord.x;
            xWeights.z = fTwo / fThree - fh * s * (fTwo - coord.x);
            xWeights.x = s * coord.x / fLambda;
            xWeights.y = fOne - xWeights.x - xWeights.z - xWeights.w;
            __private float4 yWeights;
            s = fOne - coord.y;
            yWeights.w = pown(s,3) / fLambda;
            s = coord.y * coord.y;
            yWeights.z = fTwo / fThree - fh * s * (fTwo - coord.y);
            yWeights.x = s * coord.y / fLambda;
            yWeights.y = fOne - yWeights.x - yWeights.z - yWeights.w;

            //unwrapped interpolation loop (sorry 16 iterations) 4x4 interpolation coefficients
            //y loop 0
            s = yWeights.x * (xWeights.x * target[yInterpolationIndices.x + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.x + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.x + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.x + xInterpolationIndices.w]);
            //y loop 1
            s += yWeights.y * (xWeights.x * target[yInterpolationIndices.y + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.y + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.y + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.y + xInterpolationIndices.w]);
            //y loop 2
            s += yWeights.z * (xWeights.x * target[yInterpolationIndices.z + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.z + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.z + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.z + xInterpolationIndices.w]);
            //y loop 3
            s += yWeights.w * (xWeights.x * target[yInterpolationIndices.w + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.w + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.w + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.w + xInterpolationIndices.w]);
            //now s is the value
            __private float diff = source[nIndex] - s;
            diffout[nIndex] = pown(diff,2);
            __private float Theta = yGradient[nIndex] * (float)column - xGradient[nIndex] * (float)row;
            grad0[nIndex] = diff * Theta;
            grad1[nIndex] = diff * xGradient[nIndex];
            grad2[nIndex] = diff * yGradient[nIndex];
            hessian00[nIndex] = pown(Theta,2);
            hessian01[nIndex] = Theta * xGradient[nIndex];
            hessian02[nIndex] = Theta * yGradient[nIndex];
            hessian11[nIndex] = pown(xGradient[nIndex],2);
            hessian12[nIndex] = xGradient[nIndex] * yGradient[nIndex];
            hessian22[nIndex] = pown(yGradient[nIndex],2);
        }
        else
        {
            grad0[nIndex] = fZero;
            grad1[nIndex] = fZero;
            grad2[nIndex] = fZero;
            hessian00[nIndex] = fZero;
            hessian01[nIndex] = fZero;
            hessian02[nIndex] = fZero;
            hessian11[nIndex] = fZero;
            hessian12[nIndex] = fZero;
            hessian22[nIndex] = fZero;
            diffout[nIndex] = fZero;
            mask[nIndex] = fZero;
        }
    }
}


__kernel void ftranslationErrorWithGradAndHess(const __global float *source ,const __global float *target,const __global float *xGradient,const __global float *yGradient,__global float *grad0,__global float *grad1,__global float *hessian00,__global float *hessian01,__global float *hessian11, __global float *diffout, __global float *mask, const int sourcewidth, const int sourceheight, const int targetwidth, const int targetheight, const float offsetx, const float offsety)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private float2 coord = (float2)(offsetx + ((float)column), offsety + ((float)row));

        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        __private int doubletargetwidth = 2*targetwidth;
        __private int doubletargetheight = 2*targetheight;
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            mask[nIndex] = fOne;
            //Following is the calculation using mirrored boundaries of the x and y indices of the coefficients used for interpolation
            //calculate the x coordinates for interpolation (loop unwrapped) for speed
            __private int4 xInterpolationIndices;
            __private int p = (coord.x >= 0) ? (((int)trunc(coord.x)) + 2) : (((int)trunc(coord.x)) + 1);
            //loop iteration 0
            __private int q = (p < 0) ? (-1 - p) : p;
            q = q<targetwidth?q:q%targetwidth;
            xInterpolationIndices.x = q >= targetwidth ? (targetwidth - 1 - q) : q;
            //loop iteration 1
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<targetwidth?q:q%targetwidth;
            xInterpolationIndices.y = q >= targetwidth ? (targetwidth - 1 - q) : q;
            //loop iteration 2
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<targetwidth?q:q%targetwidth;
            xInterpolationIndices.z = q >= targetwidth ? (targetwidth - 1 - q) : q;
            //loop iteration 3
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<targetwidth?q:q%targetwidth;
            xInterpolationIndices.w = q >= targetwidth ? (targetwidth - 1 - q) : q;

            __private int4 yInterpolationIndices;
            p = (coord.y >= 0) ? (((int)trunc(coord.y)) + 2) : (((int)trunc(coord.y)) + 1);
            //loop iteration 0
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.x = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 1
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.y = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 2
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.z = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 3
            p--;
            q = (p < 0) ? (-1 - p) : p;
            q = q<doubletargetheight?q:q%doubletargetheight;
            yInterpolationIndices.w = (targetheight <= q) ? (((doubletargetheight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row

            coord.x -= (coord.x >= fZero) ? (float)((int)trunc(coord.x)) : (float)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with floor
            coord.y -= (coord.y >= fZero) ? (float)((int)trunc(coord.y)) : (float)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with floor

            //Calculate the weights for interpolation
            __private float4 xWeights;
            __private float s = fOne - coord.x;
            xWeights.w = pown(s,3) / fLambda;
            s = coord.x * coord.x;
            xWeights.z = fTwo / fThree - fh * s * (fTwo - coord.x);
            xWeights.x = s * coord.x / fLambda;
            xWeights.y = fOne - xWeights.x - xWeights.z - xWeights.w;
            __private float4 yWeights;
            s = fOne - coord.y;
            yWeights.w = pown(s,3) / fLambda;
            s = coord.y * coord.y;
            yWeights.z = fTwo / fThree - fh * s * (fTwo - coord.y);
            yWeights.x = s * coord.y / fLambda;
            yWeights.y = fOne - yWeights.x - yWeights.z - yWeights.w;

            //unwrapped interpolation loop (sorry 16 iterations) 4x4 interpolation coefficients
            //y loop 0
            s = yWeights.x * (xWeights.x * target[yInterpolationIndices.x + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.x + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.x + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.x + xInterpolationIndices.w]);
            //y loop 1
            s += yWeights.y * (xWeights.x * target[yInterpolationIndices.y + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.y + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.y + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.y + xInterpolationIndices.w]);
            //y loop 2
            s += yWeights.z * (xWeights.x * target[yInterpolationIndices.z + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.z + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.z + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.z + xInterpolationIndices.w]);
            //y loop 3
            s += yWeights.w * (xWeights.x * target[yInterpolationIndices.w + xInterpolationIndices.x]
                + xWeights.y * target[yInterpolationIndices.w + xInterpolationIndices.y]
                + xWeights.z * target[yInterpolationIndices.w + xInterpolationIndices.z]
                + xWeights.w * target[yInterpolationIndices.w + xInterpolationIndices.w]);
            //now s is the value
            __private float diff = source[nIndex] - s;
            diffout[nIndex] = pown(diff,2);
            grad0[nIndex] = diff * xGradient[nIndex];
            grad1[nIndex] = diff * yGradient[nIndex];
            hessian00[nIndex] = pown(xGradient[nIndex],2);
            hessian01[nIndex] = xGradient[nIndex] * yGradient[nIndex];
            hessian11[nIndex] = pown(yGradient[nIndex],2);
        }
        else
        {
            grad0[nIndex] = fZero;
            grad1[nIndex] = fZero;
            hessian00[nIndex] = fZero;
            hessian01[nIndex] = fZero;
            hessian11[nIndex] = fZero;
            diffout[nIndex] = fZero;
            mask[nIndex] = fZero;
        }
    }
}

__kernel void fsumInLocalMemory(__global float *gdata, __local volatile float* ldata, const int size)
{
    /*
    Only a single workgroup must be started but that won't be tested here.
    The number of threads MUST be at least (size + (size % 2))/2!!!!!
    Watch out for the barriers. A barrier must be reached by all threads,
    therefore you can't use a global return statement to get rid of them,
    rather you just define behavior for the threads you want to be active
    and then let all threads hit the barriers.
    */
    __private int nIndex = get_local_id(0);
    __private int nrOfElems = size + (MODULO2(size));//now it's divisible by two
    __private int divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata[nIndex] + gdata[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    /*
    Unlike the examples presented by NVidia we don't have the luxury of assuming n being a power of two
    meaning for example the first step is 14 which is %2 = 0 but 14/2=7 which is %2 = 1!!!
    This forces us to check for every tree step iteration whether the step is %2 = 0
    */
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata[nIndex] = ldata[nIndex];//transfer back to global memory
    }
}

__kernel void fsumInLocalMemoryCombined(__global float *gdata0,__global float *gdata1,__global float *gdata2,__global float *gdata3,__global float *gdata4,__global float *gdata5,__global float *gdata6,__global float *gdata7,__global float *gdata8,__global float *gdata9,__global float *gdata10, __local volatile float* ldata, const int size)
{
    /*
    Only a single workgroup must be started but that won't be tested here.
    The number of threads MUST be at least (size + (size % 2))/2!!!!!
    Watch out for the barriers. A barrier must be reached by all threads,
    therefore you can't use a global return statement to get rid of them,
    rather you just define behavior for the threads you want to be active
    and then let all threads hit the barriers.
    */
    __private int nIndex = get_local_id(0);
    __private int nrOfElems = size + (MODULO2(size));//now it's divisible by two
    __private int divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata0[nIndex] + gdata0[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata0[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    /*
    Unlike the examples presented by NVidia we don't have the luxury of assuming n being a power of two
    meaning for example the first step is 14 which is %2 = 0 but 14/2=7 which is %2 = 1!!!
    This forces us to check for every tree step iteration whether the step is %2 = 0
    */
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata0[nIndex] = ldata[nIndex];//transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata1[nIndex] + gdata1[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata1[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata1[nIndex] = ldata[nIndex];//transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata2[nIndex] + gdata2[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata2[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata2[nIndex] = ldata[nIndex];//transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata3[nIndex] + gdata3[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata3[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata3[nIndex] = ldata[nIndex];//transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata4[nIndex] + gdata4[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata4[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata4[nIndex] = ldata[nIndex];//transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata5[nIndex] + gdata5[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata5[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata5[nIndex] = ldata[nIndex];//transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata6[nIndex] + gdata6[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata6[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata6[nIndex] = ldata[nIndex];//transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata7[nIndex] + gdata7[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata7[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata7[nIndex] = ldata[nIndex];//transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata8[nIndex] + gdata8[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata8[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata8[nIndex] = ldata[nIndex];//transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata9[nIndex] + gdata9[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata9[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata9[nIndex] = ldata[nIndex];//transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata10[nIndex] + gdata10[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata10[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata10[nIndex] = ldata[nIndex];//transfer back to global memory
    }
}


__kernel void ftranslationSumInLocalMemoryCombined(__global float *gdata0,__global float *gdata1,__global float *gdata2,__global float *gdata3,__global float *gdata4,__global float *gdata5,__global float *gdata6, __local volatile float* ldata, const int size)
{
    /*
    Only a single workgroup must be started but that won't be tested here.
    The number of threads MUST be at least (size + (size % 2))/2!!!!!
    Watch out for the barriers. A barrier must be reached by all threads,
    therefore you can't use a global return statement to get rid of them,
    rather you just define behavior for the threads you want to be active
    and then let all threads hit the barriers.
    */
    __private int nIndex = get_local_id(0);
    __private int nrOfElems = size + (MODULO2(size));//now it's divisible by two
    __private int divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata0[nIndex] + gdata0[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata0[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    /*
    Unlike the examples presented by NVidia we don't have the luxury of assuming n being a power of two
    meaning for example the first step is 14 which is %2 = 0 but 14/2=7 which is %2 = 1!!!
    This forces us to check for every tree step iteration whether the step is %2 = 0
    */
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata0[nIndex] = ldata[nIndex];//transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata1[nIndex] + gdata1[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata1[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata1[nIndex] = ldata[nIndex];//transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata2[nIndex] + gdata2[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata2[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata2[nIndex] = ldata[nIndex];//transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata3[nIndex] + gdata3[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata3[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata3[nIndex] = ldata[nIndex];//transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata4[nIndex] + gdata4[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata4[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata4[nIndex] = ldata[nIndex];//transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata5[nIndex] + gdata5[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata5[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata5[nIndex] = ldata[nIndex];//transfer back to global memory
    }

    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata6[nIndex] + gdata6[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata6[nIndex];
        }
    }
    //Now we need to do a tree based reduction, unfortunately we don't know the nr of loops at compile time
    barrier(CLK_LOCAL_MEM_FENCE);//Synchronize the local memory access
    //Didn't do loop unrolling for warp where it is not necessary to synchronize (SIMD synchronous) because the "warp" size is different on each architecture
    nrOfElems = divs;
    divs = (divs + (MODULO2(divs)))/2;
    while(nrOfElems >= 2)
    {
        if((nIndex < divs) && (nIndex + divs < nrOfElems))
        {
            ldata[nIndex] += ldata[nIndex + divs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        nrOfElems = divs;
        divs = (divs + (MODULO2(divs)))/2;
    }
    if(nIndex == 0)
    {
        gdata6[nIndex] = ldata[nIndex];//transfer back to global memory
    }
}

/*
The following code is inspired by the parallel sum reduction according to Brent's
theorem published by NVidia. Unlike their code this is designed to also run without
the requirement of power of two multiples (because some SIMDs seem to use 10 or 20).
This makes the code a lot less optimal, but if someone would like to write a separate
implementation for the different cases (which would be easy) feel free.
*/
__kernel void fparallelGroupedSumReduction(__global const float *gdata, __global float *godata /*only needs to be max block NUMBER (not size) in size*/, unsigned int size, __local volatile float* ldata)
{
    //To save on kernels we're fetching and summing already at the first level
    __private unsigned int nIndex = get_local_id(0);/*ID within a workgroup*/
    __private unsigned int blockSize = get_local_size(0); //this is the stride
    __private unsigned int i = get_group_id(0)*(blockSize*2) + nIndex;
    __private unsigned int gridSize = blockSize*2*get_num_groups(0);
    
    ldata[nIndex] = fZero;//prepare the local memory

    // get_num_groups(0) dynamically tunes the number of elements each thread sums by changing the gridSize
    // get_local_size(0) is then equal to the blocksize (the number of threads running within a block)
    while (i < size)
    {         
        ldata[nIndex] += gdata[i];
        // ensure the read is not out of bounds
        if (i + blockSize < size)
        {
            ldata[nIndex] += gdata[i+blockSize];  
        }
        i += gridSize;
    } 

    barrier(CLK_LOCAL_MEM_FENCE);//only need to synchronize the local memory

    //Now we still need to do a tree reduction in the local memory, but because we can't be sure that the groupsize is a power of two we have to do this the slow way
    i = (blockSize + (MODULO2(blockSize)))/2;
    while(blockSize >= 2)
    {
        if((nIndex < i) && (nIndex + i < blockSize))
        {
            ldata[nIndex] += ldata[nIndex + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        blockSize = i;
        i = (blockSize + (MODULO2(blockSize)))/2;
    }
    
    // write result for this block to global mem 
    if(nIndex == 0)
    {
        godata[get_group_id(0)] = ldata[0];//So in the end we still have max block Nr elements which need to be summed up.
    }
}

__kernel void frigidBodyErrorWithGradAndHessBrent(const __global float *source,
const __global float *target,
const __global float *xGradient,
const __global float *yGradient,
__global float *grad0,
__global float *grad1,
__global float *grad2,
__global float *hessian00,
__global float *hessian01,
__global float *hessian02,
__global float *hessian11,
__global float *hessian12,
__global float *hessian22,
__global float *diffout,
__global float *mask,
__local volatile float *lgrad0,
__local volatile float *lgrad1,
__local volatile float *lgrad2,
__local volatile float *lhessian00,
__local volatile float *lhessian01,
__local volatile float *lhessian02,
__local volatile float *lhessian11,
__local volatile float *lhessian12,
__local volatile float *lhessian22,
__local volatile float *ldiffout,
__local volatile float *lmask,
const int sourcewidth,
const int sourceheight,
const int targetwidth,
const int targetheight,
const float offsetx,
const float offsety,
const float angle,
const int doubleTargetWidth,
const int doubleTargetHeight)
{
    //Brent's theorem optimized version to reduce the following sum reduction to blockSize
    __private unsigned int nIndex = get_local_id(0);/*ID within a workgroup*/
    __private unsigned int blockSize = get_local_size(0); //this is the stride
    __private unsigned int i = get_group_id(0)*(blockSize*2) + nIndex;
    __private unsigned int gridSize = blockSize*2*get_num_groups(0);
    
    //prepare local buffers
    lgrad0[nIndex] = fZero;
    lgrad1[nIndex] = fZero;
    lgrad2[nIndex] = fZero;
    lhessian00[nIndex] = fZero;
    lhessian01[nIndex] = fZero;
    lhessian02[nIndex] = fZero;
    lhessian11[nIndex] = fZero;
    lhessian12[nIndex] = fZero;
    lhessian22[nIndex] = fZero;
    ldiffout[nIndex] = fZero;
    lmask[nIndex] = fZero;

    //These vectors remain the same during the loops
    __private float2 xvec = (float2)(cos(angle),-sin(angle));//Warning: this is not the x vector but it is the vector added in the x direction
    __private float2 yvec = (float2)(-xvec.y,xvec.x);//Warning: this is not the y vector but it is the vector added in the y direction
    while(i < sourcewidth * sourceheight)
    {
        __private int column = i % sourcewidth;
        __private int row = (i - column)/sourcewidth;
        __private float2 coord = (float2)(offsetx, offsety) + ((float)column) * xvec + ((float)row) * yvec;
        
        __private int4 xInterpolationIndices;
        __private int4 yInterpolationIndices;
        __private int4 combinedInterpolationIndices;
        __private float4 xWeights;
        __private float4 yWeights;
        __private int p;
        __private int q;
        __private float s;
        __private float diff;
        __private float Theta;
        
        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            lmask[nIndex] += fOne;
            //Following is the calculation using mirrored boundaries of the x and y indices of the coefficients used for interpolation
            //calculate the x coordinates for interpolation (loop unwrapped) for speed
            p = (coord.x >= 0) ? (((int)trunc(coord.x)) + 2) : (((int)trunc(coord.x)) + 1);
            //loop iteration 0
            /*
            q = (p < 0) ? (-1 - p) : p;
            is a diverging statement, although it incurs more operations the following equivalent will likely be faster
            */
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetWidth?q:q%doubleTargetWidth;
            xInterpolationIndices.x = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
            //loop iteration 1
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetWidth?q:q%doubleTargetWidth;
            xInterpolationIndices.y = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
            //loop iteration 2
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetWidth?q:q%doubleTargetWidth;
            xInterpolationIndices.z = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
            //loop iteration 3
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetWidth?q:q%doubleTargetWidth;
            xInterpolationIndices.w = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;

            p = (coord.y >= 0) ? (((int)trunc(coord.y)) + 2) : (((int)trunc(coord.y)) + 1);
            //loop iteration 0
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetHeight?q:q%doubleTargetHeight;
            yInterpolationIndices.x = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 1
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetHeight?q:q%doubleTargetHeight;
            yInterpolationIndices.y = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 2
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetHeight?q:q%doubleTargetHeight;
            yInterpolationIndices.z = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 3
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetHeight?q:q%doubleTargetHeight;
            yInterpolationIndices.w = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row

            coord.x -= (coord.x >= fZero) ? (float)((int)trunc(coord.x)) : (float)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with floor
            coord.y -= (coord.y >= fZero) ? (float)((int)trunc(coord.y)) : (float)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with floor

            //Calculate the weights for interpolation
            s = fOne - coord.x;
            xWeights.w = pown(s,3) / fLambda;
            s = coord.x * coord.x;
            xWeights.z = fTwo / fThree - fh * s * (fTwo - coord.x);
            xWeights.x = s * coord.x / fLambda;
            xWeights.y = fOne - xWeights.x - xWeights.z - xWeights.w;
            s = fOne - coord.y;
            yWeights.w = pown(s,3) / fLambda;
            s = coord.y * coord.y;
            yWeights.z = fTwo / fThree - fh * s * (fTwo - coord.y);
            yWeights.x = s * coord.y / fLambda;
            yWeights.y = fOne - yWeights.x - yWeights.z - yWeights.w;

            //unwrapped interpolation loop (sorry 16 iterations) 4x4 interpolation coefficients
            //y loop 0
            //broadcast then add
            combinedInterpolationIndices = (int4)(yInterpolationIndices.x) + xInterpolationIndices;
            s = yWeights.x * (xWeights.x * target[combinedInterpolationIndices.x]
                + xWeights.y * target[combinedInterpolationIndices.y]
                + xWeights.z * target[combinedInterpolationIndices.z]
                + xWeights.w * target[combinedInterpolationIndices.w]);
            //y loop 1
            combinedInterpolationIndices = (int4)(yInterpolationIndices.y) + xInterpolationIndices;
            s += yWeights.y * (xWeights.x * target[combinedInterpolationIndices.x]
                + xWeights.y * target[combinedInterpolationIndices.y]
                + xWeights.z * target[combinedInterpolationIndices.z]
                + xWeights.w * target[combinedInterpolationIndices.w]);
            //y loop 2
            combinedInterpolationIndices = (int4)(yInterpolationIndices.z) + xInterpolationIndices;
            s += yWeights.z * (xWeights.x * target[combinedInterpolationIndices.x]
                + xWeights.y * target[combinedInterpolationIndices.y]
                + xWeights.z * target[combinedInterpolationIndices.z]
                + xWeights.w * target[combinedInterpolationIndices.w]);
            //y loop 3
            combinedInterpolationIndices = (int4)(yInterpolationIndices.w) + xInterpolationIndices;
            s += yWeights.w * (xWeights.x * target[combinedInterpolationIndices.x]
                + xWeights.y * target[combinedInterpolationIndices.y]
                + xWeights.z * target[combinedInterpolationIndices.z]
                + xWeights.w * target[combinedInterpolationIndices.w]);
            //now s is the value
            diff = source[i] - s;
            ldiffout[nIndex] += pown(diff,2);
            Theta = yGradient[i] * (float)column - xGradient[i] * (float)row;
            lgrad0[nIndex] += diff * Theta;
            lgrad1[nIndex] += diff * xGradient[i];
            lgrad2[nIndex] += diff * yGradient[i];
            lhessian00[nIndex] += pown(Theta,2);
            lhessian01[nIndex] += Theta * xGradient[i];
            lhessian02[nIndex] += Theta * yGradient[i];
            lhessian11[nIndex] += pown(xGradient[i],2);
            lhessian12[nIndex] += xGradient[i] * yGradient[i];
            lhessian22[nIndex] += pown(yGradient[i],2);
        }
        // ensure the read is not out of bounds
        if(i + blockSize < sourcewidth * sourceheight)
        {
            __private int lIdx = i + blockSize;
            column = lIdx % sourcewidth;
            row = (lIdx - column)/sourcewidth;
            coord = (float2)(offsetx, offsety) + ((float)column) * xvec + ((float)row) * yvec;            
            Msk = (int2)((int)round(coord.x), (int)round(coord.y));
            if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
            {
                lmask[nIndex] += fOne;
                //Following is the calculation using mirrored boundaries of the x and y indices of the coefficients used for interpolation
                //calculate the x coordinates for interpolation (loop unwrapped) for speed
                p = (coord.x >= 0) ? (((int)trunc(coord.x)) + 2) : (((int)trunc(coord.x)) + 1);
                //loop iteration 0
                /*
                q = (p < 0) ? (-1 - p) : p;
                is a diverging statement, although it incurs more operations the following equivalent will likely be faster
                */
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetWidth)
                {
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division, it doesn't yield q
                }
                xInterpolationIndices.x = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
                //loop iteration 1
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetWidth)
                {
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division, it doesn't yield q
                }
                xInterpolationIndices.y = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
                //loop iteration 2
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetWidth)
                {
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division, it doesn't yield q
                }
                xInterpolationIndices.z = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
                //loop iteration 3
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetWidth)
                {
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division, it doesn't yield q
                }
                xInterpolationIndices.w = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;

                p = (coord.y >= 0) ? (((int)trunc(coord.y)) + 2) : (((int)trunc(coord.y)) + 1);
                //loop iteration 0
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetHeight)
                {
                    q -= (doubleTargetHeight) * (q / (doubleTargetHeight));//WARNING: this is an integer division which may not yield q
                }
                yInterpolationIndices.x = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
                //loop iteration 1
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetHeight)
                {
                    q -= (doubleTargetHeight) * (q / (doubleTargetHeight));//WARNING: this is an integer division which may not yield q
                }
                yInterpolationIndices.y = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
                //loop iteration 2
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetHeight)
                {
                    q -= (doubleTargetHeight) * (q / (doubleTargetHeight));//WARNING: this is an integer division which may not yield q
                }
                yInterpolationIndices.z = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
                //loop iteration 3
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetHeight)
                {
                    q -= (doubleTargetHeight) * (q / (doubleTargetHeight));//WARNING: this is an integer division which may not yield q
                }
                yInterpolationIndices.w = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row

                coord.x -= (coord.x >= fZero) ? (float)((int)trunc(coord.x)) : (float)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with floor
                coord.y -= (coord.y >= fZero) ? (float)((int)trunc(coord.y)) : (float)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with floor

                //Calculate the weights for interpolation
                s = fOne - coord.x;
                xWeights.w = pown(s,3) / fLambda;
                s = coord.x * coord.x;
                xWeights.z = fTwo / fThree - fh * s * (fTwo - coord.x);
                xWeights.x = s * coord.x / fLambda;
                xWeights.y = fOne - xWeights.x - xWeights.z - xWeights.w;
                s = fOne - coord.y;
                yWeights.w = pown(s,3) / fLambda;
                s = coord.y * coord.y;
                yWeights.z = fTwo / fThree - fh * s * (fTwo - coord.y);
                yWeights.x = s * coord.y / fLambda;
                yWeights.y = fOne - yWeights.x - yWeights.z - yWeights.w;

                //unwrapped interpolation loop (sorry 16 iterations) 4x4 interpolation coefficients
                //y loop 0
                //broadcast then add
                combinedInterpolationIndices = (int4)(yInterpolationIndices.x) + xInterpolationIndices;
                s = yWeights.x * (xWeights.x * target[combinedInterpolationIndices.x]
                    + xWeights.y * target[combinedInterpolationIndices.y]
                    + xWeights.z * target[combinedInterpolationIndices.z]
                    + xWeights.w * target[combinedInterpolationIndices.w]);
                //y loop 1
                combinedInterpolationIndices = (int4)(yInterpolationIndices.y) + xInterpolationIndices;
                s += yWeights.y * (xWeights.x * target[combinedInterpolationIndices.x]
                    + xWeights.y * target[combinedInterpolationIndices.y]
                    + xWeights.z * target[combinedInterpolationIndices.z]
                    + xWeights.w * target[combinedInterpolationIndices.w]);
                //y loop 2
                combinedInterpolationIndices = (int4)(yInterpolationIndices.z) + xInterpolationIndices;
                s += yWeights.z * (xWeights.x * target[combinedInterpolationIndices.x]
                    + xWeights.y * target[combinedInterpolationIndices.y]
                    + xWeights.z * target[combinedInterpolationIndices.z]
                    + xWeights.w * target[combinedInterpolationIndices.w]);
                //y loop 3
                combinedInterpolationIndices = (int4)(yInterpolationIndices.w) + xInterpolationIndices;
                s += yWeights.w * (xWeights.x * target[combinedInterpolationIndices.x]
                    + xWeights.y * target[combinedInterpolationIndices.y]
                    + xWeights.z * target[combinedInterpolationIndices.z]
                    + xWeights.w * target[combinedInterpolationIndices.w]);
                //now s is the value
                diff = source[lIdx] - s;
                ldiffout[nIndex] += pown(diff,2);
                Theta = yGradient[lIdx] * (float)column - xGradient[lIdx] * (float)row;
                lgrad0[nIndex] += diff * Theta;
                lgrad1[nIndex] += diff * xGradient[lIdx];
                lgrad2[nIndex] += diff * yGradient[lIdx];
                lhessian00[nIndex] += pown(Theta,2);
                lhessian01[nIndex] += Theta * xGradient[lIdx];
                lhessian02[nIndex] += Theta * yGradient[lIdx];
                lhessian11[nIndex] += pown(xGradient[lIdx],2);
                lhessian12[nIndex] += xGradient[lIdx] * yGradient[lIdx];
                lhessian22[nIndex] += pown(yGradient[lIdx],2);
            }
        }
        i += gridSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);//only need to synchronize the local memory
    //Now we still need to do a tree reduction in the local memory, but because we can't be sure that the groupsize is a power of two we have to do this the slow way
    i = (blockSize + (MODULO2(blockSize)))/2;
    while(blockSize >= 2)
    {
        if((nIndex < i) && (nIndex + i < blockSize))
        {
            lgrad0[nIndex] += lgrad0[nIndex + i];
            lgrad1[nIndex] += lgrad1[nIndex + i];
            lgrad2[nIndex] += lgrad2[nIndex + i];
            lhessian00[nIndex] += lhessian00[nIndex + i];
            lhessian01[nIndex] += lhessian01[nIndex + i];
            lhessian02[nIndex] += lhessian02[nIndex + i];
            lhessian11[nIndex] += lhessian11[nIndex + i];
            lhessian12[nIndex] += lhessian12[nIndex + i];
            lhessian22[nIndex] += lhessian22[nIndex + i];
            ldiffout[nIndex] += ldiffout[nIndex + i];
            lmask[nIndex] += lmask[nIndex + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        blockSize = i;
        i = (blockSize + (MODULO2(blockSize)))/2;
    }
    
    // write result for this block to global mem 
    if(nIndex == 0)
    {
        //So in the end we still have max block Nr elements which need to be summed up.
        i = get_group_id(0);
        grad0[i] = lgrad0[0];
        grad1[i] = lgrad1[0];
        grad2[i] = lgrad2[0];
        hessian00[i] = lhessian00[0];
        hessian01[i] = lhessian01[0];
        hessian02[i] = lhessian02[0];
        hessian11[i] = lhessian11[0];
        hessian12[i] = lhessian12[0];
        hessian22[i] = lhessian22[0];
        diffout[i] = ldiffout[0];
        mask[i] = lmask[0];
    }
}


__kernel void ftranslationErrorWithGradAndHessBrent(const __global float *source,
const __global float *target,
const __global float *xGradient,
const __global float *yGradient,
__global float *grad0,
__global float *grad1,
__global float *hessian00,
__global float *hessian01,
__global float *hessian11,
__global float *diffout,
__global float *mask,
__local volatile float *lgrad0,
__local volatile float *lgrad1,
__local volatile float *lhessian00,
__local volatile float *lhessian01,
__local volatile float *lhessian11,
__local volatile float *ldiffout,
__local volatile float *lmask,
const int sourcewidth,
const int sourceheight,
const int targetwidth,
const int targetheight,
const float offsetx,
const float offsety,
const int doubleTargetWidth,
const int doubleTargetHeight)
{
    //Brent's theorem optimized version to reduce the following sum reduction to blockSize
    __private unsigned int nIndex = get_local_id(0);/*ID within a workgroup*/
    __private unsigned int blockSize = get_local_size(0); //this is the stride
    __private unsigned int i = get_group_id(0)*(blockSize*2) + nIndex;
    __private unsigned int gridSize = blockSize*2*get_num_groups(0);
    
    //prepare local buffers
    lgrad0[nIndex] = fZero;
    lgrad1[nIndex] = fZero;
    lhessian00[nIndex] = fZero;
    lhessian01[nIndex] = fZero;
    lhessian11[nIndex] = fZero;
    ldiffout[nIndex] = fZero;
    lmask[nIndex] = fZero;

    while(i < sourcewidth * sourceheight)
    {
        __private int column = i % sourcewidth;
        __private int row = (i - column)/sourcewidth;
        __private float2 coord = (float2)(offsetx + ((float)column), offsety + ((float)row));
        
        __private int4 xInterpolationIndices;
        __private int4 yInterpolationIndices;
        __private int4 combinedInterpolationIndices;
        __private float4 xWeights;
        __private float4 yWeights;
        __private int p;
        __private int q;
        __private float s;
        __private float diff;
        
        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            lmask[nIndex] += fOne;
            //Following is the calculation using mirrored boundaries of the x and y indices of the coefficients used for interpolation
            //calculate the x coordinates for interpolation (loop unwrapped) for speed
            p = (coord.x >= 0) ? (((int)trunc(coord.x)) + 2) : (((int)trunc(coord.x)) + 1);
            //loop iteration 0
            /*
            q = (p < 0) ? (-1 - p) : p;
            is a diverging statement, although it incurs more operations the following equivalent will likely be faster
            */
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetWidth?q:q%doubleTargetWidth;
            xInterpolationIndices.x = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
            //loop iteration 1
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetWidth?q:q%doubleTargetWidth;
            xInterpolationIndices.y = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
            //loop iteration 2
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetWidth?q:q%doubleTargetWidth;
            xInterpolationIndices.z = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
            //loop iteration 3
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetWidth?q:q%doubleTargetWidth;
            xInterpolationIndices.w = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;

            p = (coord.y >= 0) ? (((int)trunc(coord.y)) + 2) : (((int)trunc(coord.y)) + 1);
            //loop iteration 0
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetHeight?q:q%doubleTargetHeight;
            yInterpolationIndices.x = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 1
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetHeight?q:q%doubleTargetHeight;
            yInterpolationIndices.y = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 2
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetHeight?q:q%doubleTargetHeight;
            yInterpolationIndices.z = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
            //loop iteration 3
            p--;
            q = abs(p) - rotate(p&(int)0x80000000,(int)1);
            q = q<doubleTargetHeight?q:q%doubleTargetHeight;
            yInterpolationIndices.w = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row

            coord.x -= (coord.x >= fZero) ? (float)((int)trunc(coord.x)) : (float)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with floor
            coord.y -= (coord.y >= fZero) ? (float)((int)trunc(coord.y)) : (float)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with floor

            //Calculate the weights for interpolation
            s = fOne - coord.x;
            xWeights.w = pown(s,3) / fLambda;
            s = coord.x * coord.x;
            xWeights.z = fTwo / fThree - fh * s * (fTwo - coord.x);
            xWeights.x = s * coord.x / fLambda;
            xWeights.y = fOne - xWeights.x - xWeights.z - xWeights.w;
            s = fOne - coord.y;
            yWeights.w = pown(s,3) / fLambda;
            s = coord.y * coord.y;
            yWeights.z = fTwo / fThree - fh * s * (fTwo - coord.y);
            yWeights.x = s * coord.y / fLambda;
            yWeights.y = fOne - yWeights.x - yWeights.z - yWeights.w;

            //unwrapped interpolation loop (sorry 16 iterations) 4x4 interpolation coefficients
            //y loop 0
            //broadcast then add
            combinedInterpolationIndices = (int4)(yInterpolationIndices.x) + xInterpolationIndices;
            s = yWeights.x * (xWeights.x * target[combinedInterpolationIndices.x]
                + xWeights.y * target[combinedInterpolationIndices.y]
                + xWeights.z * target[combinedInterpolationIndices.z]
                + xWeights.w * target[combinedInterpolationIndices.w]);
            //y loop 1
            combinedInterpolationIndices = (int4)(yInterpolationIndices.y) + xInterpolationIndices;
            s += yWeights.y * (xWeights.x * target[combinedInterpolationIndices.x]
                + xWeights.y * target[combinedInterpolationIndices.y]
                + xWeights.z * target[combinedInterpolationIndices.z]
                + xWeights.w * target[combinedInterpolationIndices.w]);
            //y loop 2
            combinedInterpolationIndices = (int4)(yInterpolationIndices.z) + xInterpolationIndices;
            s += yWeights.z * (xWeights.x * target[combinedInterpolationIndices.x]
                + xWeights.y * target[combinedInterpolationIndices.y]
                + xWeights.z * target[combinedInterpolationIndices.z]
                + xWeights.w * target[combinedInterpolationIndices.w]);
            //y loop 3
            combinedInterpolationIndices = (int4)(yInterpolationIndices.w) + xInterpolationIndices;
            s += yWeights.w * (xWeights.x * target[combinedInterpolationIndices.x]
                + xWeights.y * target[combinedInterpolationIndices.y]
                + xWeights.z * target[combinedInterpolationIndices.z]
                + xWeights.w * target[combinedInterpolationIndices.w]);
            //now s is the value
            diff = source[i] - s;
            ldiffout[nIndex] += pown(diff,2);
            lgrad0[nIndex] += diff * xGradient[i];
            lgrad1[nIndex] += diff * yGradient[i];
            lhessian00[nIndex] += pown(xGradient[i],2);
            lhessian01[nIndex] += xGradient[i] * yGradient[i];
            lhessian11[nIndex] += pown(yGradient[i],2);
        }
        // ensure the read is not out of bounds
        if(i + blockSize < sourcewidth * sourceheight)
        {
            __private int lIdx = i + blockSize;
            column = lIdx % sourcewidth;
            row = (lIdx - column)/sourcewidth;
            coord = (float2)(offsetx + ((float)column), offsety + ((float)row));
            Msk = (int2)((int)round(coord.x), (int)round(coord.y));
            if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
            {
                lmask[nIndex] += fOne;
                //Following is the calculation using mirrored boundaries of the x and y indices of the coefficients used for interpolation
                //calculate the x coordinates for interpolation (loop unwrapped) for speed
                p = (coord.x >= 0) ? (((int)trunc(coord.x)) + 2) : (((int)trunc(coord.x)) + 1);
                //loop iteration 0
                /*
                q = (p < 0) ? (-1 - p) : p;
                is a diverging statement, although it incurs more operations the following equivalent will likely be faster
                */
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetWidth)
                {
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division, it doesn't yield q
                }
                xInterpolationIndices.x = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
                //loop iteration 1
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetWidth)
                {
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division, it doesn't yield q
                }
                xInterpolationIndices.y = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
                //loop iteration 2
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetWidth)
                {
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division, it doesn't yield q
                }
                xInterpolationIndices.z = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
                //loop iteration 3
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetWidth)
                {
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division, it doesn't yield q
                }
                xInterpolationIndices.w = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;

                p = (coord.y >= 0) ? (((int)trunc(coord.y)) + 2) : (((int)trunc(coord.y)) + 1);
                //loop iteration 0
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetHeight)
                {
                    q -= (doubleTargetHeight) * (q / (doubleTargetHeight));//WARNING: this is an integer division which may not yield q
                }
                yInterpolationIndices.x = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
                //loop iteration 1
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetHeight)
                {
                    q -= (doubleTargetHeight) * (q / (doubleTargetHeight));//WARNING: this is an integer division which may not yield q
                }
                yInterpolationIndices.y = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
                //loop iteration 2
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetHeight)
                {
                    q -= (doubleTargetHeight) * (q / (doubleTargetHeight));//WARNING: this is an integer division which may not yield q
                }
                yInterpolationIndices.z = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row
                //loop iteration 3
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetHeight)
                {
                    q -= (doubleTargetHeight) * (q / (doubleTargetHeight));//WARNING: this is an integer division which may not yield q
                }
                yInterpolationIndices.w = (targetheight <= q) ? (((doubleTargetHeight) - 1 - q) * targetwidth) : (q * targetwidth);//this is the linear absolute index NOT the row

                coord.x -= (coord.x >= fZero) ? (float)((int)trunc(coord.x)) : (float)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with floor
                coord.y -= (coord.y >= fZero) ? (float)((int)trunc(coord.y)) : (float)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with floor

                //Calculate the weights for interpolation
                s = fOne - coord.x;
                xWeights.w = pown(s,3) / fLambda;
                s = coord.x * coord.x;
                xWeights.z = fTwo / fThree - fh * s * (fTwo - coord.x);
                xWeights.x = s * coord.x / fLambda;
                xWeights.y = fOne - xWeights.x - xWeights.z - xWeights.w;
                s = fOne - coord.y;
                yWeights.w = pown(s,3) / fLambda;
                s = coord.y * coord.y;
                yWeights.z = fTwo / fThree - fh * s * (fTwo - coord.y);
                yWeights.x = s * coord.y / fLambda;
                yWeights.y = fOne - yWeights.x - yWeights.z - yWeights.w;

                //unwrapped interpolation loop (sorry 16 iterations) 4x4 interpolation coefficients
                //y loop 0
                //broadcast then add
                combinedInterpolationIndices = (int4)(yInterpolationIndices.x) + xInterpolationIndices;
                s = yWeights.x * (xWeights.x * target[combinedInterpolationIndices.x]
                    + xWeights.y * target[combinedInterpolationIndices.y]
                    + xWeights.z * target[combinedInterpolationIndices.z]
                    + xWeights.w * target[combinedInterpolationIndices.w]);
                //y loop 1
                combinedInterpolationIndices = (int4)(yInterpolationIndices.y) + xInterpolationIndices;
                s += yWeights.y * (xWeights.x * target[combinedInterpolationIndices.x]
                    + xWeights.y * target[combinedInterpolationIndices.y]
                    + xWeights.z * target[combinedInterpolationIndices.z]
                    + xWeights.w * target[combinedInterpolationIndices.w]);
                //y loop 2
                combinedInterpolationIndices = (int4)(yInterpolationIndices.z) + xInterpolationIndices;
                s += yWeights.z * (xWeights.x * target[combinedInterpolationIndices.x]
                    + xWeights.y * target[combinedInterpolationIndices.y]
                    + xWeights.z * target[combinedInterpolationIndices.z]
                    + xWeights.w * target[combinedInterpolationIndices.w]);
                //y loop 3
                combinedInterpolationIndices = (int4)(yInterpolationIndices.w) + xInterpolationIndices;
                s += yWeights.w * (xWeights.x * target[combinedInterpolationIndices.x]
                    + xWeights.y * target[combinedInterpolationIndices.y]
                    + xWeights.z * target[combinedInterpolationIndices.z]
                    + xWeights.w * target[combinedInterpolationIndices.w]);
                //now s is the value
                diff = source[lIdx] - s;
                ldiffout[nIndex] += pown(diff,2);
                lgrad0[nIndex] += diff * xGradient[lIdx];
                lgrad1[nIndex] += diff * yGradient[lIdx];
                lhessian00[nIndex] += pown(xGradient[lIdx],2);
                lhessian01[nIndex] += xGradient[lIdx] * yGradient[lIdx];
                lhessian11[nIndex] += pown(yGradient[lIdx],2);
            }
        }
        i += gridSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);//only need to synchronize the local memory
    //Now we still need to do a tree reduction in the local memory, but because we can't be sure that the groupsize is a power of two we have to do this the slow way
    i = (blockSize + (MODULO2(blockSize)))/2;
    while(blockSize >= 2)
    {
        if((nIndex < i) && (nIndex + i < blockSize))
        {
            lgrad0[nIndex] += lgrad0[nIndex + i];
            lgrad1[nIndex] += lgrad1[nIndex + i];
            lhessian00[nIndex] += lhessian00[nIndex + i];
            lhessian01[nIndex] += lhessian01[nIndex + i];
            lhessian11[nIndex] += lhessian11[nIndex + i];
            ldiffout[nIndex] += ldiffout[nIndex + i];
            lmask[nIndex] += lmask[nIndex + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        blockSize = i;
        i = (blockSize + (MODULO2(blockSize)))/2;
    }
    
    // write result for this block to global mem 
    if(nIndex == 0)
    {
        //So in the end we still have max block Nr elements which need to be summed up.
        i = get_group_id(0);
        grad0[i] = lgrad0[0];
        grad1[i] = lgrad1[0];
        hessian00[i] = lhessian00[0];
        hessian01[i] = lhessian01[0];
        hessian11[i] = lhessian11[0];
        diffout[i] = ldiffout[0];
        mask[i] = lmask[0];
    }
}