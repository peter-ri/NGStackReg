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

/*
* WARNING: the results will never be identical with the CPU version because the
* CPU version uses intermediate double arrays whereas only the actual computation
* is performed with double precision here, but then downcast to float such that the
* reiteration will only have the float values available! The !ONLY! way to avoid
* this is to actually pass the image as double array. BUT this requires a lot of
* GPRAM. Local copies are not possible because the local memory on my system is
* too small SORRY!
* The code is riddled with if's and switch statements which, due to the synchronous
* stepping, forces all other GPUs to wait until the few have done their job, but unless
* someone (or myself :) ) comes up with an intelligent boundary condition handler
* it will likely stay like this sorry.
*/

//The following prepocessor code provides some definitions to be able to run
//the kernels both on a device supporting double FPA and single FPA.
//Run one kernel after the other because there is NO way of synchronizing all
//of the global memory for all workers (only in the group) except for queuing
//one kernel after the other
//Kernel verified in float mode

//TODO: MIN_SIZE is defined as 24 => any input data will never be smaller than 12 => a lot of multimirrored boundary conditions may be omitted => less branching!

/*
Various constants were put into the constant memory zone because on some GPUs the
number of registers is quite limited (even on high end hardware). Using
preprocessor constants can then lead to very strange errors (mostly out of mem).
*/

#define MODULO2(num) (num & 1)

#ifdef USE_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define FPT double
#define UCAST (double)
#define DCAST (float)
__constant FPT h = 0.5;
__constant FPT Zero = 0.0;
__constant FPT Lambda = 6.0;
__constant FPT Pole = -0.26794919243112270647255365849413;
__constant FPT One = 1.0;
__constant FPT Two = 2.0;
__constant FPT h0D3 = 0.66666666666666666666666666666666666666666666666666666666666666667;
__constant FPT h1D3 = 0.16666666666666666666666666666666666666666666666666666666666666667;
#define FPTTWO double2
#define FPTTHREE double3
#define FPTFOUR double4
#define FPTEIGHT double8
__constant FPT Three = 3.0;
#else
#define FPT float
#define UCAST
#define DCAST
__constant FPT h = 0.5f;
__constant FPT Zero = 0.0f;
__constant FPT Lambda = 6.0f;
__constant FPT Pole = -0.26794919243112270647255365849413f;
__constant FPT One = 1.0f;
__constant FPT Two = 2.0f;
__constant FPT h0D3 = 0.66666666666666666666666666666666666666666666666666666666666666667f;
__constant FPT h1D3 = 0.16666666666666666666666666666666666666666666666666666666666666667f;
#define FPTTWO float2
#define FPTTHREE float3
#define FPTFOUR float4
#define FPTEIGHT float8
__constant FPT Three = 3.0f;
#endif

//Utility functions
static inline int4 calculatexInterpolationIndxs(const FPT coordx, const int doubleTargetWidth, const int targetwidth)
{
    //Following is the calculation using mirrored boundaries of the x indices of the coefficients used for interpolation
    __private int4 xInterpolationIndices;
    __private int p = (coordx >= 0) ? (((int)trunc(coordx)) + 2) : (((int)trunc(coordx)) + 1);
    /*
    q = (p < 0) ? (-1 - p) : p;
    is a diverging statement, although it incurs more operations the following equivalent will likely be faster
    */
    __private int q = abs(p) - rotate(p&(int)0x80000000,(int)1);
    //loop iteration 0
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
    return xInterpolationIndices;
}

static inline int4 calculateyInterpolationIndxs(const FPT coordy, const int doubleTargetHeight, const int targetheight, const int targetwidth)
{
    //Following is the calculation using mirrored boundaries of the y indices of the coefficients used for interpolation
    __private int4 yInterpolationIndices;
    __private int p = (coordy >= 0) ? (((int)trunc(coordy)) + 2) : (((int)trunc(coordy)) + 1);
    //loop iteration 0
    __private int q = abs(p) - rotate(p&(int)0x80000000,(int)1);
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
    return yInterpolationIndices;
}

//Conversions as implemented in 10.1109/83.650848 (more precise)
//I read somewhere that the branch predictor typically assumes that an if statement is usually true, therefore put the most likely code in a true if statement
__kernel void CubicBSplinePrefilter2Dpremulhp(__global FPT *image /* in global space */, const int size)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the offset
    if(nIndex < size)
    {
        image[nIndex] *= Lambda;
    }
}

__kernel void TargetedCubicBSplinePrefilter2Dpremulhp(__global FPT *image /* in global space */, __global FPT *target, const int size)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the offset
    if(nIndex < size)
    {
        target[nIndex] = image[nIndex] * Lambda;
    }
}

__kernel void CubicBSplinePrefilter2DXhp(__global FPT *image /* in global space */, const int width, const int height)
{
    //high precision but slower
    __private int nIndex = get_global_id(0);//this directly corresponds to the row!!!
    if(nIndex < height)
    {
        __global FPT *prow = image + (nIndex * width);
    
        //causal initialization
        __private FPT z1 = Pole;
        __private FPT zn = pown(Pole,width);
        //__private FPT Sum = (One + Pole) * ( prow[0] + zn * prow[width - 1]);
        __private FPT Sum = (One + Pole) * ( fma(zn, prow[width - 1], prow[0]) );
        zn *= zn;
        for(int k = 1;k < width - 1; k++)
        {
            z1 *= Pole;
            zn /= Pole;
            //Sum += (z1 + zn) * prow[k];
            Sum = fma((z1 + zn), prow[k], Sum);
        }
        prow[0] = (Sum / (One - pown(Pole, 2 * width)));

        //Causal recursion
        for(int k = 1; k < width; k++)
        {
            //prow[k] = prow[k] + Pole *  prow[k-1];
            prow[k] = fma(Pole,  prow[k-1], prow[k]);
        }
        //anticausal initialization
        prow[width - 1] = (Pole * prow[width - 1] / (Pole - One));

        //Anticausal recursion
        for(int k = width - 2; 0 <= k; k--)
        {
            prow[k] = Pole * (prow[k+1] -  prow[k]);
        }
    }
}

__kernel void CubicBSplinePrefilter2DYhp(__global FPT *image /* in global space */, const int width, const int height)
{
    //high precision but slower
    __private int nIndex = get_global_id(0);//this directly corresponds to the column!!!
    if(nIndex < width)
    {
        __global FPT *prow = image + nIndex;
    
        //causal initialization
        __private FPT z1 = Pole;
        __private FPT zn = pown(Pole,height);
        //__private FPT Sum = (One + Pole) * ( prow[0] + zn *  prow[(height - 1)*width]);
        __private FPT Sum = (One + Pole) * ( fma(zn,  prow[(height - 1)*width], prow[0]));
        zn *= zn;
        for(int k = 1;k < height - 1; k++)
        {
            z1 *= Pole;
            zn /= Pole;
            //Sum += (z1 + zn) *  prow[k * width];
            Sum = fma((z1 + zn),  prow[k * width], Sum);
        }
        prow[0] = (Sum / (One - pown(Pole, 2 * height)));

        //Causal recursion
        for(int k = 1; k < height; k++)
        {
            //prow[k * width] =  prow[k * width] + Pole *  prow[(k-1)*width];
            prow[k * width] =  fma(Pole, prow[(k-1)*width], prow[k * width]);
        }

        //anticausal initialization
        prow[(height - 1)*width] = (Pole * prow[(height - 1)*width] / (Pole - One));

        //Anticausal recursion
        for(int k = height - 2; 0 <= k; k--)
        {
            prow[k*width] = (Pole * ( prow[(k+1)*width] -  prow[k*width]));
        }
    }    
}

/*
*   The problem with the following two functions is that the original program handles this by intermittently
*   copying the row or column (for x this is not a problem, no in place manipulation) but for the
*   subsequent Y run this would result in an in-place modification interfering with
*   kernels running in parallel so one has to use an intermediate storage for x and only
*   then calculate y to the final target (unfortunately this is a waste of memory
*   but I see no obvious solution to this)
*/
__kernel void BasicToCardinal2DXhp(__global FPT *image /* in global space */,__global FPT *target/* in global space */, const int width, const int height)
{
    /*
    *   This is a FIR filter so the calculations can be run simultaneously for each !TARGET! pixel
    *   Thus one also has to think the other way around not source -> target but
    *   rather target <- source
    */
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear pixel address
    if(nIndex < width*height)
    {
        //calculate the current column
        __private int col = nIndex % width;
        __private int row = (nIndex - col)/width;
        __private FPTTWO params = (FPTTWO)(h0D3, h1D3);
        __private FPTTWO imgData;
        //symmetricFirMirrorOffBounds1D
        if(col > 0 && col < (width-1))
        {
            //most common case
            //target[nIndex] = h0D3 * image[nIndex] + h1D3 * (image[nIndex - 1] + image[nIndex + 1]);
            //target[nIndex] = fma(h0D3, image[nIndex], h1D3 * (image[nIndex - 1] + image[nIndex + 1]));
            //This could be considered a dot product
            imgData = (FPTTWO)(image[nIndex], (image[nIndex - 1] + image[nIndex + 1]));
        }
        else if(col == (width-1))
        {
            /* nIndex is already row*width+width-1 so we need not waste calculation power to get this number again */
            //target[nIndex] = h0D3 * image[nIndex] + h1D3 * (image[nIndex-1] + image[nIndex]);
            //target[nIndex] = fma(h0D3, image[nIndex], h1D3 * (image[nIndex-1] + image[nIndex]));
            //This could be considered a dot product
            imgData = (FPTTWO)(image[nIndex], (image[nIndex-1] + image[nIndex]));
        }
        else
        {
            //col == 0
            /* nIndex is already row*width+0 so we need not waste calculation power to get this number again */
            //target[nIndex] = h0D3 * image[nIndex] + h1D3 * (image[nIndex] + image[nIndex+1]);
            //target[nIndex] = fma(h0D3, image[nIndex], h1D3 * (image[nIndex] + image[nIndex+1]));
            //This could be considered a dot product
            imgData = (FPTTWO)(image[nIndex], (image[nIndex] + image[nIndex+1]));
        }
        target[nIndex] = dot(params, imgData);
    }
}
__kernel void BasicToCardinal2DYhp(__global FPT *image /* in global space */,__global FPT *target/* in global space */, const int width, const int height)
{
    /*
    *   This is a FIR filter so the calculations can be run simultaneously for each !TARGET! pixel
    *   Thus one also has to think the other way around not source -> target but
    *   rather target <- source
    */
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear pixel address
    if(nIndex < width*height)
    {
        //calculate the current column
        __private int col = nIndex % width;
        __private int row = (nIndex - col)/width;
        __private FPTTWO params = (FPTTWO)(h0D3, h1D3);
        __private FPTTWO imgData;
        //symmetricFirMirrorOffBounds1D
        if(row > 0 && row < (height-1))
        {
            //most common case
            //target[nIndex] = h0D3 * image[nIndex] + h1D3 * (image[(row-1)*width+col] + image[(row+1)*width+col]);
            //target[nIndex] = fma(h0D3, image[nIndex], h1D3 * (image[(row-1)*width+col] + image[(row+1)*width+col]));
            //This could be considered a dot product
            imgData = (FPTTWO)(image[nIndex], (image[(row-1)*width+col] + image[(row+1)*width+col]));
        }
        else if(row == (height-1))
        {
            /* nIndex is already (height-1)*width+col so we need not waste calculation power to get this number again */
            //target[nIndex] = h0D3 * image[nIndex] + h1D3 * (image[(row-1)*width+col] + image[nIndex]);
            //target[nIndex] = fma(h0D3, image[nIndex], h1D3 * (image[(row-1)*width+col] + image[nIndex]));
            //This could be considered a dot product
            imgData = (FPTTWO)(image[nIndex], (image[(row-1)*width+col] + image[nIndex]));
        }
        else
        {
            //row == 0
            /* nIndex is already col so we need not waste calculation power to get this number again */
            //target[nIndex] = h0D3 * image[nIndex] + h1D3 * (image[nIndex] + image[width+col]);
            //target[nIndex] = fma(h0D3, image[nIndex], h1D3 * (image[nIndex] + image[width+col]));
            //This could be considered a dot product
            imgData = (FPTTWO)(image[nIndex], (image[nIndex] + image[width+col]));
        }
        target[nIndex] = dot(params, imgData);
    }
}
#ifdef USE_DOUBLE
__constant const FPT Z0 = -0.5352804307964381655424037816816460718339231523426924148812;
__constant const FPT Z1 = -0.122554615192326690515272264359357343605486549427295558490763;
__constant const FPT Z2 = -0.0091486948096082769285930216516478534156925639545994482648003;
__constant const FPT Lambda7 = 5040.0;
__constant const FPT h0D7 = 0.4793650793650793650793650793650793650793650793650793650793650793651;
__constant const FPT h1D7 = 0.23630952380952380952380952380952380952380952380952380952380952380952;
__constant const FPT h2D7 = 0.023809523809523809523809523809523809523809523809523809523809523810;
__constant const FPT h3D7 = 0.00019841269841269841269841269841269841269841269841269841269841269841;
__constant const FPTFOUR hD7vec = (FPTFOUR)(0.4793650793650793650793650793650793650793650793650793650793650793651, 0.23630952380952380952380952380952380952380952380952380952380952380952, 0.023809523809523809523809523809523809523809523809523809523809523810, 0.00019841269841269841269841269841269841269841269841269841269841269841);
#else
__constant const FPT Z0 = -0.5352804307964381655424037816816460718339231523426924148812f;
__constant const FPT Z1 = -0.122554615192326690515272264359357343605486549427295558490763f;
__constant const FPT Z2 = -0.0091486948096082769285930216516478534156925639545994482648003f;
__constant const FPT Lambda7 = 5040.0f;
__constant const FPT h0D7 = 0.4793650793650793650793650793650793650793650793650793650793650793651f;
__constant const FPT h1D7 = 0.23630952380952380952380952380952380952380952380952380952380952380952f;
__constant const FPT h2D7 = 0.023809523809523809523809523809523809523809523809523809523809523810f;
__constant const FPT h3D7 = 0.00019841269841269841269841269841269841269841269841269841269841269841f;
__constant const FPTFOUR hD7vec = (FPTFOUR)(0.4793650793650793650793650793650793650793650793650793650793650793651f, 0.23630952380952380952380952380952380952380952380952380952380952380952f, 0.023809523809523809523809523809523809523809523809523809523809523810f, 0.00019841269841269841269841269841269841269841269841269841269841269841f);
#endif


__kernel void CubicBSplinePrefilter2DDeg7premulhp(__global FPT *image /* in global space */, const int size)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the offset
    if(nIndex < size)
    {
        image[nIndex] *= Lambda7;
    }
}

__kernel void CubicBSplinePrefilter2DXDeg7hp(__global FPT *image /* in global space */, const int width, const int height)
{
    //high precision but slower
    __private int nIndex = get_global_id(0);//this directly corresponds to the row!!!
    if(nIndex < height)
    {
        __global FPT *prow = image + (nIndex * width);
        //For beta 7th order this has to be done 3 times
        //Iteration 1
        //causal initialization
        __private FPT z1 = Z0;
        __private FPT zn = pown(Z0,width);
        //__private FPT Sum = (One + Z0) * ( prow[0] + zn *  prow[width - 1]);
        __private FPT Sum = (One + Z0) * ( fma(zn,  prow[width - 1], prow[0]));
        zn *= zn;
        for(int k = 1;k < width - 1; k++)
        {
            z1 *= Z0;
            zn /= Z0;
            Sum += (z1 + zn) *  prow[k];
        }
        prow[0] = (Sum / (One - pown(Z0, 2 * width)));
    
        //Causal recursion
        for(int k = 1; k < width; k++)
        {
            //prow[k] += Z0 *  prow[k-1];
            prow[k] = fma(Z0,  prow[k-1], prow[k]);
        }
        //anticausal initialization
        prow[width - 1] = Z0 * prow[width - 1] / (Z0 - One);

        //Anticausal recursion
        for(int k = width - 2; 0 <= k; k--)
        {
            prow[k] = Z0 * (prow[k+1] -  prow[k]);
        }
        //Iteration 2
        //causal initialization
        z1 = Z1;
        zn = pown(Z1,width);
        //Sum = (One + Z1) * ( prow[0] + zn *  prow[width - 1]);
        Sum = (One + Z1) * ( fma(zn, prow[width - 1], prow[0]));
        zn *= zn;
        for(int k = 1;k < width - 1; k++)
        {
            z1 *= Z1;
            zn /= Z1;
            //Sum += (z1 + zn) *  prow[k];
            Sum = fma((z1 + zn), prow[k], Sum);
        }
        prow[0] = (Sum / (One - pown(Z1, 2 * width)));
    
        //Causal recursion
        for(int k = 1; k < width; k++)
        {
            //prow[k] += Z1 *  prow[k-1];
            prow[k] = fma(Z1, prow[k-1], prow[k]);
        }
        //anticausal initialization
        prow[width - 1] = (Z1 * prow[width - 1] / (Z1 - One));

        //Anticausal recursion
        for(int k = width - 2; 0 <= k; k--)
        {
            prow[k] = (Z1 * ( prow[k+1] -  prow[k]));
        }
        //Iteration 3
        //causal initialization
        z1 = Z2;
        zn = pown(Z2,width);
        //Sum = (One + Z2) * ( prow[0] + zn *  prow[width - 1]);
        Sum = (One + Z2) * (fma(zn, prow[width - 1], prow[0]));
        zn *= zn;
        for(int k = 1;k < width - 1; k++)
        {
            z1 *= Z2;
            zn /= Z2;
            //Sum += (z1 + zn) *  prow[k];
            Sum = fma((z1 + zn),  prow[k], Sum);
        }
        prow[0] = (Sum / (One - pown(Z2, 2 * width)));
    
        //Causal recursion
        for(int k = 1; k < width; k++)
        {
            //prow[k] = prow[k] + Z2 * prow[k-1];
            prow[k] = fma(Z2, prow[k-1], prow[k]);
        }
        //anticausal initialization
        prow[width - 1] = (Z2 *  prow[width - 1] / (Z2 - One));

        //Anticausal recursion
        for(int k = width - 2; 0 <= k; k--)
        {
            prow[k] = Z2 * ( prow[k+1] -  prow[k]);
        }
    }
}

__kernel void CubicBSplinePrefilter2DYDeg7hp(__global FPT *image /* in global space */, const int width, const int height)
{
    //high precision but slower
    __private int nIndex = get_global_id(0);//this directly corresponds to the column!!!
    if(nIndex < width)
    {
        __global FPT *prow = image + nIndex;
        //For beta 7th order this has to be done 3 times
        //Iteration 1
        //causal initialization
        __private FPT z1 = Z0;
        __private FPT zn = pown(Z0,height);
        //__private FPT Sum = (One + Z0) * ( prow[0] + zn *  prow[(height - 1)*width]);
        __private FPT Sum = (One + Z0) * ( fma(zn, prow[(height - 1)*width], prow[0]) );
        zn *= zn;
        for(int k = 1;k < height - 1; k++)
        {
            z1 *= Z0;
            zn /= Z0;
            //Sum += (z1 + zn) *  prow[k*width];
            Sum = fma((z1 + zn), prow[k*width], Sum);
        }
        prow[0] = (Sum / (One - pown(Z0, 2 * height)));
    
        //Causal recursion
        for(int k = 1; k < height; k++)
        {
            //prow[k*width] += Z0 *  prow[(k-1)*width];
            prow[k*width] = fma(Z0,  prow[(k-1)*width], prow[k*width]);
        }
        //anticausal initialization
        prow[(height - 1)*width] = Z0 *  prow[(height - 1)*width] / (Z0 - One);

        //Anticausal recursion
        for(int k = height - 2; 0 <= k; k--)
        {
            prow[k*width] = Z0 * ( prow[(k+1)*width] -  prow[k*width]);
        }
        //Iteration 2
        //causal initialization
        z1 = Z1;
        zn = pown(Z1,height);
        //Sum = (One + Z1) * ( prow[0] + zn *  prow[(height - 1)*width]);
        Sum = (One + Z1) * ( fma(zn, prow[(height - 1)*width],prow[0]) );
        zn *= zn;
        for(int k = 1;k < height - 1; k++)
        {
            z1 *= Z1;
            zn /= Z1;
            //Sum += (z1 + zn) *  prow[k*width];
            Sum = fma((z1 + zn),  prow[k*width], Sum);
        }
        prow[0] = (Sum / (One - pown(Z1, 2 * height)));
    
        //Causal recursion
        for(int k = 1; k < height; k++)
        {
            prow[k*width] += Z1 *  prow[(k-1)*width];
        }
        //anticausal initialization
        prow[(height - 1)*width] = Z1 *  prow[(height - 1)*width] / (Z1 - One);

        //Anticausal recursion
        for(int k = height - 2; 0 <= k; k--)
        {
            prow[k*width] = Z1 * ( prow[(k+1)*width] -  prow[k*width]);
        }
        //Iteration 3
        //causal initialization
        z1 = Z2;
        zn = pown(Z2,height);
        //Sum = (One + Z2) * ( prow[0] + zn *  prow[(height - 1)*width]);
        Sum = (One + Z2) * ( fma(zn, prow[(height - 1)*width], prow[0]) );
        zn *= zn;
        for(int k = 1;k < height - 1; k++)
        {
            z1 *= Z2;
            zn /= Z2;
            //Sum += (z1 + zn) *  prow[k*width];
            Sum = fma((z1 + zn),  prow[k*width], Sum);
        }
        prow[0] = (Sum / (One - pown(Z2, 2 * height)));
    
        //Causal recursion
        for(int k = 1; k < height; k++)
        {
            //prow[k*width] += Z2 *  prow[(k-1)*width];
            prow[k*width] = fma(Z2, prow[(k-1)*width], prow[k*width]);
        }
        //anticausal initialization
        prow[(height - 1)*width] = (Z2 *  prow[(height - 1)*width] / (Z2 - One));

        //Anticausal recursion
        for(int k = height - 2; 0 <= k; k--)
        {
            prow[k*width] = Z2 * ( prow[(k+1)*width] -  prow[k*width]);
        }
    }
}

/*
*   The problem with the following two functions is that the original program handles this by intermittently
*   copying the row or column (for x this is not a problem, no in place manipulation) but for the
*   subsequent Y run this would result in an in-place modification interfering with
*   kernels running in parallel so one has to use an intermediate storage for x and only
*   then calculate y to the final target (unfortunately this is a waste of memory
*   but I see no obvious solution to this)
*/
__kernel void BasicToCardinal2DXhpDeg7(__global FPT *image /* in global space */,__global FPT *target/* in global space */, const int width, const int height)
{
    /*
    *   This is a FIR filter so the calculations can be run simultaneously for each !TARGET! pixel
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
        __private FPTFOUR imgData;
        __private FPTFOUR imgData2;
        if(col > 2 && col < (width-3))
        {
            //most common case
            //target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex-1] + image[nIndex+1]) + h2D7 * (image[nIndex-2] + image[nIndex+2]) + h3D7 * (image[nIndex-3] + image[nIndex+3]);
            //This could be considered a dot product
            imgData = (FPTFOUR)(image[nIndex], image[nIndex-1], image[nIndex-2], image[nIndex-3]);
            imgData2 = (FPTFOUR)(Zero, image[nIndex+1], image[nIndex+2], image[nIndex+3]);
            imgData += imgData2;
        }
        else if(col == 1)
        {
            //target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex-1] + image[nIndex+1]) + h2D7 * (image[nIndex-1] + image[nIndex+2]) + h3D7 * (image[nIndex] + image[nIndex+3]);
            //This could be considered a dot product
            imgData = (FPTFOUR)(image[nIndex], image[nIndex-1], image[nIndex-1], image[nIndex]);
            imgData2 = (FPTFOUR)(Zero, image[nIndex+1], image[nIndex+2], image[nIndex+3]);
            imgData += imgData2;
        }
        else if(col == 2)
        {
            //target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex-1] + image[nIndex+1]) + h2D7 * (image[nIndex-2] + image[nIndex+2]) + h3D7 * (image[nIndex-2] + image[nIndex+3]);
            //This could be considered a dot product
            imgData = (FPTFOUR)(image[nIndex], image[nIndex-1], image[nIndex-2], image[nIndex-2]);
            imgData2 = (FPTFOUR)(Zero, image[nIndex+1], image[nIndex+2], image[nIndex+3]);
            imgData += imgData2;
        }
        else if(col == (width-3))
        {
            /* nIndex is already row*width+width-3 so we need not waste calculation power to get this number again */
            //target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex-1] + image[nIndex+1]) + h2D7 * (image[nIndex-2] + image[nIndex+2]) + h3D7 * (image[nIndex-3] + image[nIndex+2]);
            //This could be considered a dot product
            imgData = (FPTFOUR)(image[nIndex], image[nIndex-1], image[nIndex-2], image[nIndex-3]);
            imgData2 = (FPTFOUR)(Zero, image[nIndex+1], image[nIndex+2], image[nIndex+2]);
            imgData += imgData2;
        }
        else if(col == (width-2))
        {
            /* nIndex is already row*width+width-2 so we need not waste calculation power to get this number again */
            //target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex-1] + image[nIndex+1]) + h2D7 * (image[nIndex-2] + image[nIndex+1]) + h3D7 * (image[nIndex-3] + image[nIndex]);
            //This could be considered a dot product
            imgData = (FPTFOUR)(image[nIndex], image[nIndex-1], image[nIndex-2], image[nIndex-3]);
            imgData2 = (FPTFOUR)(Zero, image[nIndex+1], image[nIndex+1], image[nIndex]);
            imgData += imgData2;
        }
        else if(col == (width-1))
        {
            /* nIndex is already row*width+width-1 so we need not waste calculation power to get this number again */
            //target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex-1] + image[nIndex]) + h2D7 * (image[nIndex-2] + image[nIndex-1]) + h3D7 * (image[nIndex-3] + image[nIndex-2]);
            //This could be considered a dot product
            imgData = (FPTFOUR)(image[nIndex], image[nIndex-1], image[nIndex-2], image[nIndex-3]);
            imgData2 = (FPTFOUR)(Zero, image[nIndex], image[nIndex-1], image[nIndex-2]);
            imgData += imgData2;
        }
        else
        {
            //col == 0
            /* nIndex is already row*width+0 so we need not waste calculation power to get this number again */
            //target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex] + image[nIndex+1]) + h2D7 * (image[nIndex+1] + image[nIndex+2])+ h3D7 * (image[nIndex+2] + image[nIndex+3]);
            //This could be considered a dot product
            imgData = (FPTFOUR)(image[nIndex], image[nIndex], image[nIndex+1], image[nIndex+2]);
            imgData2 = (FPTFOUR)(Zero, image[nIndex+1], image[nIndex+2], image[nIndex+3]);
            imgData += imgData2;
        }
        target[nIndex] = dot(hD7vec, imgData);
    }
}
__kernel void BasicToCardinal2DYhpDeg7(__global FPT *image /* in global space */,__global FPT *target/* in global space */, const int width, const int height)
{
    /*
    *   This is a FIR filter so the calculations can be run simultaneously for each !TARGET! pixel
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
        //TODO: could use vector math to calculate the indices
        __private FPTFOUR imgData;
        __private FPTFOUR imgData2;
        if(row > 2 && row < (height-3))
        {
            //most common case
            //target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex-width] + image[nIndex+width]) + h2D7 * (image[nIndex-2*width] + image[nIndex+2*width]) + h3D7 * (image[nIndex-3*width] + image[nIndex+3*width]);
            //This could be considered a dot product
            imgData = (FPTFOUR)(image[nIndex], image[nIndex-width], image[nIndex-2*width], image[nIndex-3*width]);
            imgData2 = (FPTFOUR)(Zero, image[nIndex+width], image[nIndex+2*width], image[nIndex+3*width]);
            imgData += imgData2;
        }
        else if(row == 1)
        {
            //target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[col] + image[2*width+col]) + h2D7 * (image[col] + image[3*width+col]) + h3D7 * (image[nIndex] + image[4*width+col]);
            //This could be considered a dot product
            imgData = (FPTFOUR)(image[nIndex], image[col], image[col], image[nIndex]);
            imgData2 = (FPTFOUR)(Zero, image[2*width+col], image[3*width+col], image[4*width+col]);
            imgData += imgData2;
        }
        else if(row == 2)
        {
            //target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[width+col] + image[3*width+col]) + h2D7 * (image[col] + image[4*width+col]) + h3D7 * (image[col] + image[5*width+col]);
            //This could be considered a dot product
            imgData = (FPTFOUR)(image[nIndex], image[width+col], image[col], image[col]);
            imgData2 = (FPTFOUR)(Zero, image[3*width+col], image[4*width+col], image[5*width+col]);
            imgData += imgData2;
        }
        else if(row == (height-3))
        {
            /* nIndex is already row*width+width-3 so we need not waste calculation power to get this number again */
            //target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex-width] + image[nIndex+width]) + h2D7 * (image[nIndex-2*width] + image[nIndex+2*width]) + h3D7 * (image[nIndex-3*width] + image[nIndex+2*width]);
            //This could be considered a dot product
            imgData = (FPTFOUR)(image[nIndex], image[nIndex-width], image[nIndex-2*width], image[nIndex-3*width]);
            imgData2 = (FPTFOUR)(Zero, image[nIndex+width], image[nIndex+2*width], image[nIndex+2*width]);
            imgData += imgData2;
        }
        else if(row == (height-2))
        {
            /* nIndex is already row*width+width-2 so we need not waste calculation power to get this number again */
            //target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex-width] + image[nIndex+width]) + h2D7 * (image[nIndex-2*width] + image[nIndex+2*width]) + h3D7 * (image[nIndex-3*width] + image[nIndex]);
            //This could be considered a dot product
            imgData = (FPTFOUR)(image[nIndex], image[nIndex-width], image[nIndex-2*width], image[nIndex-3*width]);
            imgData2 = (FPTFOUR)(Zero, image[nIndex+width], image[nIndex+2*width], image[nIndex]);
            imgData += imgData2;
        }
        else if(row == (height-1))
        {
            /* nIndex is already row*width+width-1 so we need not waste calculation power to get this number again */
            //target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex-width] + image[nIndex]) + h2D7 * (image[nIndex-2*width] + image[nIndex-width]) + h3D7 * (image[nIndex-3*width] + image[nIndex-3*width]);
            //This could be considered a dot product
            imgData = (FPTFOUR)(image[nIndex], image[nIndex-width], image[nIndex-2*width], image[nIndex-3*width]);
            imgData2 = (FPTFOUR)(Zero, image[nIndex], image[nIndex-width], image[nIndex-3*width]);
            imgData += imgData2;
        }
        else
        {
            //row == 0
            /* nIndex is already row*width+0 so we need not waste calculation power to get this number again */
            //target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex] + image[width+col]) + h2D7 * (image[width+col] + image[2*width+col])+ h3D7 * (image[2*width+col] + image[3*width+col]);
            //This could be considered a dot product
            imgData = (FPTFOUR)(image[nIndex], image[nIndex], image[width+col], image[2*width+col]);
            imgData2 = (FPTFOUR)(Zero, image[width+col], image[2*width+col], image[3*width+col]);
            imgData += imgData2;
        }
        target[nIndex] = dot(hD7vec, imgData);
    }
}

/*
*   The following kernels perform the scale down for the construction of the L_2 pyramid
*   for this a "demi"half image is needed, because the X function only halves the width
*   and only then the image height is halved in the Y function. As already mentioned above
*   try to think target <- source and not source -> target
*/
#ifdef USE_DOUBLE
__constant const FPT rh0 = 0.375;
__constant const FPT rh1 = 0.25;
__constant const FPT rh2 = 0.0625;
__constant const FPTTHREE rhvec = (FPTTHREE)(0.375, 0.25, 0.0625);
#else
__constant const FPT rh0 = 0.375f;
__constant const FPT rh1 = 0.25f;
__constant const FPT rh2 = 0.0625f;
__constant const FPTTHREE rhvec = (FPTTHREE)(0.375f, 0.25f, 0.0625f);
#endif
__kernel void reduceDual1DX(__global FPT *image /* in global space */,__global FPT *target/* in global space */, const int width, const int height, const int halfwidth)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear pixel address in the !TARGET! image
    __private int col = nIndex % halfwidth;
    __private int row = (nIndex - col)/halfwidth;
    //reduceDual1D
    if(nIndex < halfwidth*height)
    {
        /*
        *   Warning: be aware that the coordinate systems of the target and the source image are different now
        *   because the width of the target is only roughly half of that of the source image. The row
        *   is the same though so calculate everything from the corresponding row offset not nIndex
        */
        //halfwidth >= 2 is guaranteed
        __private FPTTHREE imgData;
        __private FPTTHREE imgData2;
        if(col > 0 && col < (halfwidth - 1))
        {
            //most common case
            //target[nIndex] = rh0 * image[row*width + col*2] + rh1 * (image[row*width + col*2 - 1] + image[row*width + col*2 + 1]) + rh2 * (image[row*width + col*2 - 2] + image[row*width + col*2 + 2]);
            imgData = (FPTTHREE)(image[row*width + col*2], image[row*width + col*2 - 1], image[row*width + col*2 - 2]);
            imgData2 = (FPTTHREE)(Zero, image[row*width + col*2 + 1], image[row*width + col*2 + 2]);
            imgData += imgData2;
        }
        else if(col == halfwidth - 1)
        {
            if(width == (2 * halfwidth))//Yes this can be different if width % 2 != 0
            {
                //target[nIndex] = rh0 * image[row*width+width-2] + rh1 * (image[row*width+width-3] + image[row*width+width-1]) + rh2 * (image[row*width+width-4] + image[row*width+width-1]);
                imgData = (FPTTHREE)(image[row*width+width-2], image[row*width+width-3], image[row*width+width-4]);
                imgData2 = (FPTTHREE)(Zero, image[row*width+width-1], image[row*width+width-1]);
                imgData += imgData2;
            }
            else
            {
                //target[nIndex] = rh0 * image[row*width+width-3] + rh1 * (image[row*width+width-4] + image[row*width+width-2]) + rh2 * (image[row*width+width-5] + image[row*width+width-1]);
                imgData = (FPTTHREE)(image[row*width+width-3], image[row*width+width-4], image[row*width+width-5]);
                imgData2 = (FPTTHREE)(Zero, image[row*width+width-2], image[row*width+width-1]);
                imgData += imgData2;
            }
        }
        else
        {
            //col == 0
            //target[nIndex] =  rh0 * image[row*width] + rh1 * (image[row*width] + image[row*width+1]) + rh2 * (image[row*width+1] + image[row*width+2]);
            imgData = (FPTTHREE)(image[row*width], image[row*width], image[row*width+1]);
            imgData2 = (FPTTHREE)(Zero, image[row*width+1], image[row*width+2]);
            imgData += imgData2;
        }
        target[nIndex] = dot(rhvec, imgData);
    }
}

__kernel void reduceDual1DY(__global FPT *image /* in global space */,__global FPT *target/* in global space */, const int halfwidth, const int height, const int halfheight)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear pixel address in the !TARGET! image
    __private int col = nIndex % halfwidth;
    __private int row = (nIndex - col)/halfwidth;
    //reduceDual1D
    if(nIndex < halfwidth*halfheight)
    {
        /*
        *   Warning: be aware that the coordinate systems of the target and the source image are different now
        *   because the width and height of the target are only roughly half of that of the source image. The col
        *   is the same for the Y version
        */
        //halfheight >= 2 is guaranteed
        __private FPTTHREE imgData;
        __private FPTTHREE imgData2;
        if(row > 0 && row < (halfheight - 1))
        {
            //most common case
            //target[nIndex] = rh0 * image[2*row*halfwidth+col] + rh1 * (image[(2*row - 1)*halfwidth+col] + image[(2*row + 1)*halfwidth+col]) + rh2 * (image[(2*row - 2)*halfwidth+col] + image[(2*row + 2)*halfwidth+col]);
            imgData = (FPTTHREE)(image[2*row*halfwidth+col], image[(2*row - 1)*halfwidth+col], image[(2*row - 2)*halfwidth+col]);
            imgData2 = (FPTTHREE)(Zero, image[(2*row + 1)*halfwidth+col], image[(2*row + 2)*halfwidth+col]);
            imgData += imgData2;                
        }
        else if(row == halfheight - 1)
        {
            if(height == (2 * halfheight))//Yes this can be different if height % 2 != 0
            {
                //target[nIndex] = rh0 * image[(height - 2)*halfwidth+col] + rh1 * (image[(height - 3)*halfwidth+col] + image[(height - 1)*halfwidth+col]) + rh2 * (image[(height - 4)*halfwidth+col] + image[(height - 1)*halfwidth+col]);
                imgData = (FPTTHREE)(image[(height - 2)*halfwidth+col], image[(height - 3)*halfwidth+col], image[(height - 4)*halfwidth+col]);
                imgData2 = (FPTTHREE)(Zero, image[(height - 1)*halfwidth+col], image[(height - 1)*halfwidth+col]);
                imgData += imgData2;
            }
            else
            {
                //target[nIndex] = rh0 * image[(height - 3)*halfwidth+col] + rh1 * (image[(height - 4)*halfwidth+col] + image[(height - 2)*halfwidth+col]) + rh2 * (image[(height - 5)*halfwidth+col] + image[(height - 1)*halfwidth+col]);
                imgData = (FPTTHREE)(image[(height - 3)*halfwidth+col], image[(height - 4)*halfwidth+col], image[(height - 5)*halfwidth+col]);
                imgData2 = (FPTTHREE)(Zero, image[(height - 2)*halfwidth+col], image[(height - 1)*halfwidth+col]);
                imgData += imgData2;
            }
        }
        else
        {
            //row == 0
            //target[nIndex] =  rh0 * image[col] + rh1 * (image[col] + image[halfwidth+col]) + rh2 * (image[halfwidth+col] + image[2*halfwidth+col]);
            imgData = (FPTTHREE)(image[col], image[col], image[halfwidth+col]);
            imgData2 = (FPTTHREE)(Zero, image[halfwidth+col], image[2*halfwidth+col]);
            imgData += imgData2;
        }
        target[nIndex] = dot(rhvec, imgData);
    }
}


/*
*   The following functions are for generating the derivatives of the B-splines (from the coefficients)
*/
__kernel void antiSymmetricFirMirrorOffBounds1DX(__global FPT *image ,__global FPT *target , const int width, const int height)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !TARGET! pixel
    if(nIndex < height * width)
    {
        __private int column = nIndex % width;
        __private int row = (nIndex - column)/width;
        if(column > 0 && column < (width - 1))
        {
            //most common case
            target[nIndex] = h * (image[nIndex + 1] - image[nIndex - 1]);
        }
        else if(column == width - 1)
        {
            target[nIndex] = h * (image[nIndex] - image[nIndex - 1]);
        }
        else
        {
            //column == 0
            target[nIndex] = h * (image[nIndex + 1] - image[nIndex]);
        }
    }
}

__kernel void antiSymmetricFirMirrorOffBounds1DY(__global FPT *image ,__global FPT *target , const int width, const int height)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !TARGET! pixel
    if(nIndex < height * width)
    {
        __private int column = nIndex % width;
        __private int row = (nIndex - column)/width;
        if(row > 0 && row < (height - 1))
        {
            //most common case
            target[nIndex] = h * (image[(row+1)*width+column] - image[(row-1)*width+column]);
        }
        else if(row == height - 1)
        {
            target[nIndex] = h * (image[nIndex] - image[(row-1)*width+column]);
        }
        else
        {
            //row == 0
            target[nIndex] = h * (image[width+column] - image[nIndex]);
        }
    }
}

static inline FPT interpolate(const FPTTWO coordinates, const int4 xInterpolationIndxs, const int4 yInterpolationIndxs, __global const FPT* target)
{
    __private FPTTWO coord;
    __private FPTTWO unusedCoordFloor;
    //coord.x -= (coord.x >= Zero) ? (FPT)((int)trunc(coord.x)) : (FPT)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with floor
    //coord.y -= (coord.y >= Zero) ? (FPT)((int)trunc(coord.y)) : (FPT)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with floor
    coord = fract(coordinates, &unusedCoordFloor);

    //Calculate the weights for interpolation
    __private FPTFOUR xWeights;
    __private FPTFOUR yWeights;
    __private FPT s = One - coord.x;
    
    xWeights.w = pown(s,3) / Lambda;
    s = coord.x * coord.x;
    xWeights.z = Two / Three - h * s * (Two - coord.x);
    xWeights.x = s * coord.x / Lambda;
    xWeights.y = One - xWeights.x - xWeights.z - xWeights.w;
    
    s = One - coord.y;
    yWeights.w = pown(s,3) / Lambda;
    s = coord.y * coord.y;
    yWeights.z = Two / Three - h * s * (Two - coord.y);
    yWeights.x = s * coord.y / Lambda;
    yWeights.y = One - yWeights.x - yWeights.z - yWeights.w;

    __private int4 interpolationIndices = (int4)yInterpolationIndxs.x;
    interpolationIndices += xInterpolationIndxs;
    __private FPTFOUR intermediate;
    __private FPTFOUR values = (FPTFOUR)(target[interpolationIndices.x],
                                         target[interpolationIndices.y],
                                         target[interpolationIndices.z],
                                         target[interpolationIndices.w]);
    intermediate.x = dot(xWeights, values);
    interpolationIndices = (int4)yInterpolationIndxs.y;
    interpolationIndices += xInterpolationIndxs;
    values = (FPTFOUR)(target[interpolationIndices.x],
                       target[interpolationIndices.y],
                       target[interpolationIndices.z],
                       target[interpolationIndices.w]);
    intermediate.y = dot(xWeights, values);
    interpolationIndices = (int4)yInterpolationIndxs.z;
    interpolationIndices += xInterpolationIndxs;
    values = (FPTFOUR)(target[interpolationIndices.x],
                       target[interpolationIndices.y],
                       target[interpolationIndices.z],
                       target[interpolationIndices.w]);
    intermediate.z = dot(xWeights, values);
    interpolationIndices = (int4)yInterpolationIndxs.w;
    interpolationIndices += xInterpolationIndxs;
    values = (FPTFOUR)(target[interpolationIndices.x],
                       target[interpolationIndices.y],
                       target[interpolationIndices.z],
                       target[interpolationIndices.w]);
    intermediate.w = dot(xWeights, values);
    s = dot(yWeights, intermediate);
    //now s is the value
    return s;
}

__kernel void sumInLocalMemory(__global FPT *gdata, __local volatile FPT* ldata, const int size)
{
    /*
    Only a single workgroup must be started but this won't be tested here.
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

/*
The following code is inspired by the parallel sum reduction according to Brent's
theorem published by NVidia. Unlike their code this is designed to also run without
the requirement of power of two multiples (because some SIMDs seem to use 10 or 20).
This makes the code a lot less optimal, but if someone would like to write a separate
implementation for the different cases (which would be easy) feel free.
*/
__kernel void parallelGroupedSumReduction(__global const FPT *gdata, __global FPT *godata /*only needs to be max block NUMBER (not size) in size*/, unsigned int size, __local volatile FPT* ldata)
{
    //To save on kernels we're fetching and summing already at the first level
    __private unsigned int nIndex = get_local_id(0);/*ID within a workgroup*/
    __private unsigned int blockSize = get_local_size(0); //this is the stride
    __private unsigned int i = get_group_id(0)*(blockSize*2) + nIndex;
    __private unsigned int gridSize = blockSize*2*get_num_groups(0);
    
    ldata[nIndex] = Zero;//prepare the local memory

    // get_num_groups(0) dynamically tunes the number of elements each thread sums by changing the gridSize
    // get_local_size(0) is then equal to the blocksize (the number of threads running within a block)
    while (i < size)
    {         
        ldata[nIndex] += gdata[i];
        // ensure reads are not out of bounds
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

//calculate the square error
__kernel void affineError(const __global FPT *source,
const __global FPT *target, 
__global FPT *diffout, 
__global FPT *mask, 
const int sourcewidth, 
const int sourceheight, 
const int targetwidth, 
const int targetheight, 
const int doubletargetwidth, 
const int doubletargetheight, 
const FPT offsetx, 
const FPT offsety, 
const FPT a11, 
const FPT a12, 
const FPT a21, 
const FPT a22)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;
        
        __private FPTTWO xvec = (FPTTWO)(a11,a21);//warning: this is not the x vector but it is the vector added in the x direction
        __private FPTTWO yvec = (FPTTWO)(a12,a22);//warning: this is not the y vector but it is the vector added in the y direction
        __private FPTTWO coord = (FPTTWO)(offsetx, offsety) + ((FPT)column) * xvec + ((FPT)row) * yvec;

        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        __private int4 xInterpolationIndxs;
        __private int4 yInterpolationIndxs;
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            mask[nIndex] = One;
            xInterpolationIndxs = calculatexInterpolationIndxs(coord.x, doubletargetwidth, targetwidth);
            yInterpolationIndxs = calculateyInterpolationIndxs(coord.y, doubletargetheight, targetheight, targetwidth); 
            __private FPT s = interpolate(coord, xInterpolationIndxs, yInterpolationIndxs, target);
            diffout[nIndex] = pown(source[nIndex] - s,2);
        }
        else
        {
            diffout[nIndex] = Zero;
            mask[nIndex] = Zero;
        } 
    }   
}

__kernel void affineTransformImageWithBsplineInterpolation(const __global FPT *source ,
__global FPT *target, 
const int sourcewidth, 
const int sourceheight, 
const int doubleSourceWidth, 
const int doubleSourceHeight, 
const FPT offsetx, 
const FPT offsety, 
const FPT a11, 
const FPT a12, 
const FPT a21, 
const FPT a22)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private FPTTWO xvec = (FPTTWO)(a11,a21);//warning: this is not the x vector but it is the vector added in the x direction
        __private FPTTWO yvec = (FPTTWO)(a12,a22);//warning: this is not the y vector but it is the vector added in the y direction
        __private FPTTWO coord = (FPTTWO)(offsetx, offsety) + ((FPT)column) * xvec + ((FPT)row) * yvec;

        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        __private int4 xInterpolationIndxs;
        __private int4 yInterpolationIndxs;
        if ((Msk.x >= 0) && (Msk.x < sourcewidth) && (Msk.y >= 0) && (Msk.y < sourceheight))
        {
            xInterpolationIndxs = calculatexInterpolationIndxs(coord.x, doubleSourceWidth, sourcewidth);
            yInterpolationIndxs = calculateyInterpolationIndxs(coord.y, doubleSourceHeight, sourceheight, sourcewidth);
            target[nIndex] = interpolate(coord, xInterpolationIndxs, yInterpolationIndxs, source);
        }
        else
        {
            target[nIndex] = Zero;
        } 
    }   
}

__kernel void resizingAffineTransformImageWithBsplineInterpolation(const __global FPT *source,
__global FPT *target, 
const int sourcewidth, 
const int sourceheight, 
const int doubleSourceWidth, 
const int doubleSourceHeight, 
const int targetwidth, 
const int targetheight, 
const FPT offsetx, 
const FPT offsety, 
const FPT a11, 
const FPT a12, 
const FPT a21, 
const FPT a22)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !TARGET! pixel
    if(nIndex < targetwidth * targetheight)
    {
        __private int column = nIndex % targetwidth;
        __private int row = (nIndex - column)/targetwidth;

        __private FPTTWO xvec = (FPTTWO)(a11,a21);//warning: this is not the x vector but it is the vector added in the x direction
        __private FPTTWO yvec = (FPTTWO)(a12,a22);//warning: this is not the y vector but it is the vector added in the y direction
        __private FPTTWO coord = (FPTTWO)(offsetx, offsety) + ((FPT)column) * xvec + ((FPT)row) * yvec;

        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        __private int4 xInterpolationIndxs;
        __private int4 yInterpolationIndxs;
        if ((Msk.x >= 0) && (Msk.x < sourcewidth) && (Msk.y >= 0) && (Msk.y < sourceheight))
        {
            xInterpolationIndxs = calculatexInterpolationIndxs(coord.x, doubleSourceWidth, sourcewidth);
            yInterpolationIndxs = calculateyInterpolationIndxs(coord.y, doubleSourceHeight, sourceheight, sourcewidth);
            target[nIndex] = interpolate(coord, xInterpolationIndxs, yInterpolationIndxs, source);
        }
        else
        {
            target[nIndex] = Zero;
        } 
    }   
}

