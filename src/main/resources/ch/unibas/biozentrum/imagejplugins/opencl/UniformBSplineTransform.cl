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
__constant FPT Lambda = 6.0;
__constant FPT Pole = -0.26794919243112270647255365849413;
__constant FPT One = 1.0;
__constant FPT Two = 2.0;
__constant FPT h0D3 = 0.66666666666666666666666666666666666666666666666666666666666666667;
__constant FPT h1D3 = 0.16666666666666666666666666666666666666666666666666666666666666667;
#else
#define FPT float
#define UCAST
#define DCAST
__constant FPT Lambda = 6.0f;
__constant FPT Pole = -0.26794919243112270647255365849413f;
__constant FPT One = 1.0f;
__constant FPT Two = 2.0f;
__constant FPT h0D3 = 0.66666666666666666666666666666666666666666666666666666666666666667f;
__constant FPT h1D3 = 0.16666666666666666666666666666666666666666666666666666666666666667f;
#endif

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
        __private FPT Sum = (One + Pole) * ( prow[0] + zn * prow[width - 1]);
        zn *= zn;
        for(int k = 1;k < width - 1; k++)
        {
            z1 *= Pole;
            zn /= Pole;
            Sum += (z1 + zn) * prow[k];
        }
        prow[0] = (Sum / (One - pown(Pole, 2 * width)));

        //Causal recursion
        for(int k = 1; k < width; k++)
        {
            prow[k] = prow[k] + Pole *  prow[k-1];
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
        __private FPT Sum = (One + Pole) * ( prow[0] + zn *  prow[(height - 1)*width]);
        zn *= zn;
        for(int k = 1;k < height - 1; k++)
        {
            z1 *= Pole;
            zn /= Pole;
            Sum += (z1 + zn) *  prow[k * width];
        }
        prow[0] = (Sum / (One - pown(Pole, 2 * height)));

        //Causal recursion
        for(int k = 1; k < height; k++)
        {
            prow[k * width] =  prow[k * width] + Pole *  prow[(k-1)*width];
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
        //symmetricFirMirrorOffBounds1D
        if(col > 0 && col < (width-1))
        {
            //most common case
            target[nIndex] = h0D3 * image[nIndex] + h1D3 * (image[nIndex - 1] + image[nIndex + 1]);
        }
        else if(col == (width-1))
        {
            /* nIndex is already row*width+width-1 so we need not waste calculation power to get this number again */
            target[nIndex] = h0D3 * image[nIndex] + h1D3 * (image[nIndex-1] + image[nIndex]);
        }
        else
        {
            //col == 0
            /* nIndex is already row*width+0 so we need not waste calculation power to get this number again */
            target[nIndex] = h0D3 * image[nIndex] + h1D3 * (image[nIndex] + image[nIndex+1]);
        }
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
        //symmetricFirMirrorOffBounds1D
        if(row > 0 && row < (height-1))
        {
            //most common case
            target[nIndex] = h0D3 * image[nIndex] + h1D3 * (image[(row-1)*width+col] + image[(row+1)*width+col]);
        }
        else if(row == (height-1))
        {
            /* nIndex is already (height-1)*width+col so we need not waste calculation power to get this number again */
            target[nIndex] = h0D3 * image[nIndex] + h1D3 * (image[(row-1)*width+col] + image[nIndex]);
        }
        else
        {
            //row == 0
            /* nIndex is already col so we need not waste calculation power to get this number again */
            target[nIndex] = h0D3 * image[nIndex] + h1D3 * (image[nIndex] + image[width+col]);
        }
    }
}
#ifdef USE_DOUBLE
__constant FPT Z0 = -0.5352804307964381655424037816816460718339231523426924148812;
__constant FPT Z1 = -0.122554615192326690515272264359357343605486549427295558490763;
__constant FPT Z2 = -0.0091486948096082769285930216516478534156925639545994482648003;
__constant FPT Lambda7 = 5040.0;
__constant FPT h0D7 = 0.4793650793650793650793650793650793650793650793650793650793650793651;
__constant FPT h1D7 = 0.23630952380952380952380952380952380952380952380952380952380952380952;
__constant FPT h2D7 = 0.023809523809523809523809523809523809523809523809523809523809523810;
__constant FPT h3D7 = 0.00019841269841269841269841269841269841269841269841269841269841269841;
#else
__constant FPT Z0 = -0.5352804307964381655424037816816460718339231523426924148812f;
__constant FPT Z1 = -0.122554615192326690515272264359357343605486549427295558490763f;
__constant FPT Z2 = -0.0091486948096082769285930216516478534156925639545994482648003f;
__constant FPT Lambda7 = 5040.0f;
__constant FPT h0D7 = 0.4793650793650793650793650793650793650793650793650793650793650793651f;
__constant FPT h1D7 = 0.23630952380952380952380952380952380952380952380952380952380952380952f;
__constant FPT h2D7 = 0.023809523809523809523809523809523809523809523809523809523809523810f;
__constant FPT h3D7 = 0.00019841269841269841269841269841269841269841269841269841269841269841f;
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
        __private FPT Sum = (One + Z0) * ( prow[0] + zn *  prow[width - 1]);
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
            prow[k] += Z0 *  prow[k-1];
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
        Sum = (One + Z1) * ( prow[0] + zn *  prow[width - 1]);
        zn *= zn;
        for(int k = 1;k < width - 1; k++)
        {
            z1 *= Z1;
            zn /= Z1;
            Sum += (z1 + zn) *  prow[k];
        }
        prow[0] = (Sum / (One - pown(Z1, 2 * width)));
    
        //Causal recursion
        for(int k = 1; k < width; k++)
        {
            prow[k] += Z1 *  prow[k-1];
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
        Sum = (One + Z2) * ( prow[0] + zn *  prow[width - 1]);
        zn *= zn;
        for(int k = 1;k < width - 1; k++)
        {
            z1 *= Z2;
            zn /= Z2;
            Sum += (z1 + zn) *  prow[k];
        }
        prow[0] = (Sum / (One - pown(Z2, 2 * width)));
    
        //Causal recursion
        for(int k = 1; k < width; k++)
        {
            prow[k] = prow[k] + Z2 * prow[k-1];
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
        __private FPT Sum = (One + Z0) * ( prow[0] + zn *  prow[(height - 1)*width]);
        zn *= zn;
        for(int k = 1;k < height - 1; k++)
        {
            z1 *= Z0;
            zn /= Z0;
            Sum += (z1 + zn) *  prow[k*width];
        }
        prow[0] = (Sum / (One - pown(Z0, 2 * height)));
    
        //Causal recursion
        for(int k = 1; k < height; k++)
        {
            prow[k*width] += Z0 *  prow[(k-1)*width];
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
        Sum = (One + Z1) * ( prow[0] + zn *  prow[(height - 1)*width]);
        zn *= zn;
        for(int k = 1;k < height - 1; k++)
        {
            z1 *= Z1;
            zn /= Z1;
            Sum += (z1 + zn) *  prow[k*width];
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
        Sum = (One + Z2) * ( prow[0] + zn *  prow[(height - 1)*width]);
        zn *= zn;
        for(int k = 1;k < height - 1; k++)
        {
            z1 *= Z2;
            zn /= Z2;
            Sum += (z1 + zn) *  prow[k*width];
        }
        prow[0] = (Sum / (One - pown(Z2, 2 * height)));
    
        //Causal recursion
        for(int k = 1; k < height; k++)
        {
            prow[k*width] += Z2 *  prow[(k-1)*width];
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
        if(col > 2 && col < (width-3))
        {
            //most common case
            target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex-1] + image[nIndex+1]) + h2D7 * (image[nIndex-2] + image[nIndex+2]) + h3D7 * (image[nIndex-3] + image[nIndex+3]);
        }
        else if(col == 1)
        {
            target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex-1] + image[nIndex+1]) + h2D7 * (image[nIndex-1] + image[nIndex+2]) + h3D7 * (image[nIndex] + image[nIndex+3]);
        }
        else if(col == 2)
        {
            target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex-1] + image[nIndex+1]) + h2D7 * (image[nIndex-2] + image[nIndex+2]) + h3D7 * (image[nIndex-2] + image[nIndex+3]);
        }
        else if(col == (width-3))
        {
            /* nIndex is already row*width+width-3 so we need not waste calculation power to get this number again */
            target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex-1] + image[nIndex+1]) + h2D7 * (image[nIndex-2] + image[nIndex+2]) + h3D7 * (image[nIndex-3] + image[nIndex+2]);
        }
        else if(col == (width-2))
        {
            /* nIndex is already row*width+width-2 so we need not waste calculation power to get this number again */
            target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex-1] + image[nIndex+1]) + h2D7 * (image[nIndex-2] + image[nIndex+1]) + h3D7 * (image[nIndex-3] + image[nIndex]);
        }
        else if(col == (width-1))
        {
            /* nIndex is already row*width+width-1 so we need not waste calculation power to get this number again */
            target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex-1] + image[nIndex]) + h2D7 * (image[nIndex-2] + image[nIndex-1]) + h3D7 * (image[nIndex-3] + image[nIndex-2]);
        }
        else
        {
            //col == 0
            /* nIndex is already row*width+0 so we need not waste calculation power to get this number again */
            target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex] + image[nIndex+1]) + h2D7 * (image[nIndex+1] + image[nIndex+2])+ h3D7 * (image[nIndex+2] + image[nIndex+3]);
        }
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
        if(row > 2 && row < (height-3))
        {
            //most common case
            target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex-width] + image[nIndex+width]) + h2D7 * (image[nIndex-2*width] + image[nIndex+2*width]) + h3D7 * (image[nIndex-3*width] + image[nIndex+3*width]);
        }
        else if(row == 1)
        {
            target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[col] + image[2*width+col]) + h2D7 * (image[col] + image[3*width+col]) + h3D7 * (image[nIndex] + image[4*width+col]);
        }
        else if(row == 2)
        {
            target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[width+col] + image[3*width+col]) + h2D7 * (image[col] + image[4*width+col]) + h3D7 * (image[col] + image[5*width+col]);
        }
        else if(row == (height-3))
        {
            /* nIndex is already row*width+width-3 so we need not waste calculation power to get this number again */
            target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex-width] + image[nIndex+width]) + h2D7 * (image[nIndex-2*width] + image[nIndex+2*width]) + h3D7 * (image[nIndex-3*width] + image[nIndex+2*width]);
        }
        else if(row == (height-2))
        {
            /* nIndex is already row*width+width-2 so we need not waste calculation power to get this number again */
            target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex-width] + image[nIndex+width]) + h2D7 * (image[nIndex-2*width] + image[nIndex+2*width]) + h3D7 * (image[nIndex-3*width] + image[nIndex]);
        }
        else if(row == (height-1))
        {
            /* nIndex is already row*width+width-1 so we need not waste calculation power to get this number again */
            target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex-width] + image[nIndex]) + h2D7 * (image[nIndex-2*width] + image[nIndex-width]) + h3D7 * (image[nIndex-3*width] + image[nIndex-3*width]);
        }
        else
        {
            //row == 0
            /* nIndex is already row*width+0 so we need not waste calculation power to get this number again */
            target[nIndex] = h0D7 * image[nIndex] + h1D7 * (image[nIndex] + image[width+col]) + h2D7 * (image[width+col] + image[2*width+col])+ h3D7 * (image[2*width+col] + image[3*width+col]);
        }
    }
}

/*
*   The following kernels perform the scale down for the construction of the L_2 pyramid
*   for this a "demi"half image is needed, because the X function only halves the width
*   and only then the image height is halved in the Y function. As already mentioned above
*   try to think target <- source and not source -> target
*/
#ifdef USE_DOUBLE
__constant FPT rh0 = 0.375;
__constant FPT rh1 = 0.25;
__constant FPT rh2 = 0.0625;
#else
__constant FPT rh0 = 0.375f;
__constant FPT rh1 = 0.25f;
__constant FPT rh2 = 0.0625f;
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
        if(col > 0 && col < (halfwidth - 1))
        {
            //most common case
            target[nIndex] = rh0 * image[row*width + col*2] + rh1 * (image[row*width + col*2 - 1] + image[row*width + col*2 + 1]) + rh2 * (image[row*width + col*2 - 2] + image[row*width + col*2 + 2]);
        }
        else if(col == halfwidth - 1)
        {
            if(width == (2 * halfwidth))//Yes this can be different if width % 2 != 0
            {
                target[nIndex] = rh0 * image[row*width+width-2] + rh1 * (image[row*width+width-3] + image[row*width+width-1]) + rh2 * (image[row*width+width-4] + image[row*width+width-1]);
            }
            else
            {
                target[nIndex] = rh0 * image[row*width+width-3] + rh1 * (image[row*width+width-4] + image[row*width+width-2]) + rh2 * (image[row*width+width-5] + image[row*width+width-1]);
            }
        }
        else
        {
            //col == 0
            target[nIndex] =  rh0 * image[row*width] + rh1 * (image[row*width] + image[row*width+1]) + rh2 * (image[row*width+1] + image[row*width+2]);
        }
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
        if(row > 0 && row < (halfheight - 1))
        {
            //most common case
            target[nIndex] = rh0 * image[2*row*halfwidth+col] + rh1 * (image[(2*row - 1)*halfwidth+col] + image[(2*row + 1)*halfwidth+col]) + rh2 * (image[(2*row - 2)*halfwidth+col] + image[(2*row + 2)*halfwidth+col]);                
        }
        else if(row == halfheight - 1)
        {
            if(height == (2 * halfheight))//Yes this can be different if height % 2 != 0
            {
                target[nIndex] = rh0 * image[(height - 2)*halfwidth+col] + rh1 * (image[(height - 3)*halfwidth+col] + image[(height - 1)*halfwidth+col]) + rh2 * (image[(height - 4)*halfwidth+col] + image[(height - 1)*halfwidth+col]);
            }
            else
            {
                target[nIndex] = rh0 * image[(height - 3)*halfwidth+col] + rh1 * (image[(height - 4)*halfwidth+col] + image[(height - 2)*halfwidth+col]) + rh2 * (image[(height - 5)*halfwidth+col] + image[(height - 1)*halfwidth+col]);
            }
        }
        else
        {
            //row == 0
            target[nIndex] =  rh0 * image[col] + rh1 * (image[col] + image[halfwidth+col]) + rh2 * (image[halfwidth+col] + image[2*halfwidth+col]);
        }
    }
}



#ifdef USE_DOUBLE
__constant FPT h = 0.5;
__constant FPT Zero = 0.0;
#else
__constant FPT h = 0.5f;
__constant FPT Zero = 0.0f;
#endif
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

#ifdef USE_DOUBLE
#define FPTTWO double2
#define FPTFOUR double4
__constant FPT Three = 3.0;
#else
#define FPTTWO float2
#define FPTFOUR float4
__constant FPT Three = 3.0f;
#endif
//calculate the square error
__kernel void rigidBodyError(const __global FPT *source ,const __global FPT *target, __global FPT *diffout, __global FPT *mask, const int sourcewidth, const int sourceheight, const int targetwidth, const int targetheight, const FPT offsetx, const FPT offsety, const FPT angle)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private FPTTWO xvec = (FPTTWO)(cos(angle),-sin(angle));//warning: this is not the x vector but it is the vector added in the x direction
        __private FPTTWO yvec = (FPTTWO)(-xvec.y,xvec.x);//warning: this is not the y vector but it is the vector added in the y direction
        __private FPTTWO coord = (FPTTWO)(offsetx, offsety) + ((FPT)column) * xvec + ((FPT)row) * yvec;

        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        __private int doubletargetwidth = 2*targetwidth;
        __private int doubletargetheight = 2*targetheight;
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            mask[nIndex] = One;
            //Following is the calculation using mirrored boundaries of the x and y indices of the coefficients used for interpolation
            //calculate the x coordinates for interpolation (loop unwrapped) for speed
            __private int4 xInterpolationIndices;
            __private int p = (coord.x >= 0) ? (((int)trunc(coord.x)) + 2) : (((int)trunc(coord.x)) + 1);
            //loop iteration 0
            __private int q = (p < 0) ? (-1 - p) : p;
            //kick out divergence (but calculating the modulo ma actually be slower than divergence so maybe use a slightly diverging statement)
            q = q<doubletargetwidth?q:q%doubletargetwidth;
            //q %= doubletargetwidth; //will allways give the right answer
            /*
            if(q >= doubletargetwidth)
            {
                //q -= (2*targetwidth) * (q / (2*targetwidth)); //Warning: this is an integer division it doesn't yield q (in fact it is a simple modulo operation)
                
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

            coord.x -= (coord.x >= Zero) ? (FPT)((int)trunc(coord.x)) : (FPT)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with floor
            coord.y -= (coord.y >= Zero) ? (FPT)((int)trunc(coord.y)) : (FPT)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with floor

            //Calculate the weights for interpolation
            __private FPTFOUR xWeights;
            __private FPT s = One - coord.x;
            xWeights.w = pown(s,3) / Lambda;
            s = coord.x * coord.x;
            xWeights.z = Two / Three - h * s * (Two - coord.x);
            xWeights.x = s * coord.x / Lambda;
            xWeights.y = One - xWeights.x - xWeights.z - xWeights.w;
            __private FPTFOUR yWeights;
            s = One - coord.y;
            yWeights.w = pown(s,3) / Lambda;
            s = coord.y * coord.y;
            yWeights.z = Two / Three - h * s * (Two - coord.y);
            yWeights.x = s * coord.y / Lambda;
            yWeights.y = One - yWeights.x - yWeights.z - yWeights.w;

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
            diffout[nIndex] = Zero;
            mask[nIndex] = Zero;
        } 
    }   
}

//calculate the square error
__kernel void translationError(const __global FPT *source ,const __global FPT *target, __global FPT *diffout, __global FPT *mask, const int sourcewidth, const int sourceheight, const int targetwidth, const int targetheight, const FPT offsetx, const FPT offsety)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private FPTTWO coord = (FPTTWO)(offsetx + ((FPT)column), offsety + ((FPT)row));

        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        __private int doubletargetwidth = 2*targetwidth;
        __private int doubletargetheight = 2*targetheight;
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            mask[nIndex] = One;
            //Following is the calculation using mirrored boundaries of the x and y indices of the coefficients used for interpolation
            //calculate the x coordinates for interpolation (loop unwrapped) for speed
            __private int4 xInterpolationIndices;
            __private int p = (coord.x >= 0) ? (((int)trunc(coord.x)) + 2) : (((int)trunc(coord.x)) + 1);
            //loop iteration 0
            __private int q = (p < 0) ? (-1 - p) : p;
            //kick out divergence (but calculating the modulo ma actually be slower than divergence so maybe use a slightly diverging statement)
            q = q<doubletargetwidth?q:q%doubletargetwidth;
            //q %= doubletargetwidth; //will allways give the right answer
            /*
            if(q >= doubletargetwidth)
            {
                //q -= (2*targetwidth) * (q / (2*targetwidth)); //Warning: this is an integer division it doesn't yield q (in fact it is a simple modulo operation)
                
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

            coord.x -= (coord.x >= Zero) ? (FPT)((int)trunc(coord.x)) : (FPT)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with floor
            coord.y -= (coord.y >= Zero) ? (FPT)((int)trunc(coord.y)) : (FPT)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with floor

            //Calculate the weights for interpolation
            __private FPTFOUR xWeights;
            __private FPT s = One - coord.x;
            xWeights.w = pown(s,3) / Lambda;
            s = coord.x * coord.x;
            xWeights.z = Two / Three - h * s * (Two - coord.x);
            xWeights.x = s * coord.x / Lambda;
            xWeights.y = One - xWeights.x - xWeights.z - xWeights.w;
            __private FPTFOUR yWeights;
            s = One - coord.y;
            yWeights.w = pown(s,3) / Lambda;
            s = coord.y * coord.y;
            yWeights.z = Two / Three - h * s * (Two - coord.y);
            yWeights.x = s * coord.y / Lambda;
            yWeights.y = One - yWeights.x - yWeights.z - yWeights.w;

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
            diffout[nIndex] = Zero;
            mask[nIndex] = Zero;
        } 
    }   
}

__kernel void rigidBodyErrorWithGradAndHess(const __global FPT *source ,const __global FPT *target,const __global FPT *xGradient,const __global FPT *yGradient,__global FPT *grad0,__global FPT *grad1,__global FPT *grad2,__global FPT *hessian00,__global FPT *hessian01,__global FPT *hessian02,__global FPT *hessian11,__global FPT *hessian12,__global FPT *hessian22, __global FPT *diffout, __global FPT *mask, const int sourcewidth, const int sourceheight, const int targetwidth, const int targetheight, const FPT offsetx, const FPT offsety, const FPT angle)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private FPTTWO xvec = (FPTTWO)(cos(angle),-sin(angle));//warning: this is not the x vector but it is the vector added in the x direction
        __private FPTTWO yvec = (FPTTWO)(-xvec.y,xvec.x);//warning: this is not the y vector but it is the vector added in the y direction
        __private FPTTWO coord = (FPTTWO)(offsetx, offsety) + ((FPT)column) * xvec + ((FPT)row) * yvec;

        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        __private int doubletargetwidth = 2*targetwidth;
        __private int doubletargetheight = 2*targetheight;
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            mask[nIndex] = One;
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

            coord.x -= (coord.x >= Zero) ? (FPT)((int)trunc(coord.x)) : (FPT)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with floor
            coord.y -= (coord.y >= Zero) ? (FPT)((int)trunc(coord.y)) : (FPT)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with floor

            //Calculate the weights for interpolation
            __private FPTFOUR xWeights;
            __private FPT s = One - coord.x;
            xWeights.w = pown(s,3) / Lambda;
            s = coord.x * coord.x;
            xWeights.z = Two / Three - h * s * (Two - coord.x);
            xWeights.x = s * coord.x / Lambda;
            xWeights.y = One - xWeights.x - xWeights.z - xWeights.w;
            __private FPTFOUR yWeights;
            s = One - coord.y;
            yWeights.w = pown(s,3) / Lambda;
            s = coord.y * coord.y;
            yWeights.z = Two / Three - h * s * (Two - coord.y);
            yWeights.x = s * coord.y / Lambda;
            yWeights.y = One - yWeights.x - yWeights.z - yWeights.w;

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
            __private FPT diff = source[nIndex] - s;
            diffout[nIndex] = pown(diff,2);
            __private FPT Theta = yGradient[nIndex] * (FPT)column - xGradient[nIndex] * (FPT)row;
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
            grad0[nIndex] = Zero;
            grad1[nIndex] = Zero;
            grad2[nIndex] = Zero;
            hessian00[nIndex] = Zero;
            hessian01[nIndex] = Zero;
            hessian02[nIndex] = Zero;
            hessian11[nIndex] = Zero;
            hessian12[nIndex] = Zero;
            hessian22[nIndex] = Zero;
            diffout[nIndex] = Zero;
            mask[nIndex] = Zero;
        }
    }
}


__kernel void translationErrorWithGradAndHess(const __global FPT *source ,const __global FPT *target,const __global FPT *xGradient,const __global FPT *yGradient,__global FPT *grad0,__global FPT *grad1,__global FPT *hessian00,__global FPT *hessian01,__global FPT *hessian11, __global FPT *diffout, __global FPT *mask, const int sourcewidth, const int sourceheight, const int targetwidth, const int targetheight, const FPT offsetx, const FPT offsety)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private FPTTWO coord = (FPTTWO)(offsetx + ((FPT)column), offsety + ((FPT)row));

        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        __private int doubletargetwidth = 2*targetwidth;
        __private int doubletargetheight = 2*targetheight;
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            mask[nIndex] = One;
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

            coord.x -= (coord.x >= Zero) ? (FPT)((int)trunc(coord.x)) : (FPT)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with floor
            coord.y -= (coord.y >= Zero) ? (FPT)((int)trunc(coord.y)) : (FPT)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with floor

            //Calculate the weights for interpolation
            __private FPTFOUR xWeights;
            __private FPT s = One - coord.x;
            xWeights.w = pown(s,3) / Lambda;
            s = coord.x * coord.x;
            xWeights.z = Two / Three - h * s * (Two - coord.x);
            xWeights.x = s * coord.x / Lambda;
            xWeights.y = One - xWeights.x - xWeights.z - xWeights.w;
            __private FPTFOUR yWeights;
            s = One - coord.y;
            yWeights.w = pown(s,3) / Lambda;
            s = coord.y * coord.y;
            yWeights.z = Two / Three - h * s * (Two - coord.y);
            yWeights.x = s * coord.y / Lambda;
            yWeights.y = One - yWeights.x - yWeights.z - yWeights.w;

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
            __private FPT diff = source[nIndex] - s;
            diffout[nIndex] = pown(diff,2);
            grad0[nIndex] = diff * xGradient[nIndex];
            grad1[nIndex] = diff * yGradient[nIndex];
            hessian00[nIndex] = pown(xGradient[nIndex],2);
            hessian01[nIndex] = xGradient[nIndex] * yGradient[nIndex];
            hessian11[nIndex] = pown(yGradient[nIndex],2);
        }
        else
        {
            grad0[nIndex] = Zero;
            grad1[nIndex] = Zero;
            hessian00[nIndex] = Zero;
            hessian01[nIndex] = Zero;
            hessian11[nIndex] = Zero;
            diffout[nIndex] = Zero;
            mask[nIndex] = Zero;
        }
    }
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

__kernel void sumInLocalMemoryCombined(__global FPT *gdata0,__global FPT *gdata1,__global FPT *gdata2,__global FPT *gdata3,__global FPT *gdata4,__global FPT *gdata5,__global FPT *gdata6,__global FPT *gdata7,__global FPT *gdata8,__global FPT *gdata9,__global FPT *gdata10, __local volatile FPT* ldata, const int size)
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

__kernel void translationSumInLocalMemoryCombined(__global FPT *gdata0,__global FPT *gdata1,__global FPT *gdata2,__global FPT *gdata3,__global FPT *gdata4,__global FPT *gdata5,__global FPT *gdata6, __local volatile FPT* ldata, const int size)
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


__kernel void rigidBodyErrorWithGradAndHessBrent(const __global FPT *source,
const __global FPT *target,
const __global FPT *xGradient,
const __global FPT *yGradient,
__global FPT *grad0,
__global FPT *grad1,
__global FPT *grad2,
__global FPT *hessian00,
__global FPT *hessian01,
__global FPT *hessian02,
__global FPT *hessian11,
__global FPT *hessian12,
__global FPT *hessian22,
__global FPT *diffout,
__global FPT *mask,
__local volatile FPT *lgrad0,
__local volatile FPT *lgrad1,
__local volatile FPT *lgrad2,
__local volatile FPT *lhessian00,
__local volatile FPT *lhessian01,
__local volatile FPT *lhessian02,
__local volatile FPT *lhessian11,
__local volatile FPT *lhessian12,
__local volatile FPT *lhessian22,
__local volatile FPT *ldiffout,
__local volatile FPT *lmask,
const int sourcewidth,
const int sourceheight,
const int targetwidth,
const int targetheight,
const FPT offsetx,
const FPT offsety,
const FPT angle,
const int doubleTargetWidth,
const int doubleTargetHeight)
{
    //Brent's theorem optimized version to reduce the following sum reduction to blockSize
    __private unsigned int nIndex = get_local_id(0);/*ID within a workgroup*/
    __private unsigned int blockSize = get_local_size(0); //this is the stride
    __private unsigned int i = get_group_id(0)*(blockSize*2) + nIndex;
    __private unsigned int gridSize = blockSize*2*get_num_groups(0);
    
    //prepare local buffers
    lgrad0[nIndex] = Zero;
    lgrad1[nIndex] = Zero;
    lgrad2[nIndex] = Zero;
    lhessian00[nIndex] = Zero;
    lhessian01[nIndex] = Zero;
    lhessian02[nIndex] = Zero;
    lhessian11[nIndex] = Zero;
    lhessian12[nIndex] = Zero;
    lhessian22[nIndex] = Zero;
    ldiffout[nIndex] = Zero;
    lmask[nIndex] = Zero;

    //These vectors remain the same during the loops
    __private FPTTWO xvec = (FPTTWO)(cos(angle),-sin(angle));//warning: this is not the x vector but it is the vector added in the x direction
    __private FPTTWO yvec = (FPTTWO)(-xvec.y,xvec.x);//warning: this is not the y vector but it is the vector added in the y direction
    while(i < sourcewidth * sourceheight)
    {
        __private int column = i % sourcewidth;
        __private int row = (i - column)/sourcewidth;
        __private FPTTWO coord = (FPTTWO)(offsetx, offsety) + ((FPT)column) * xvec + ((FPT)row) * yvec;
        
        __private int4 xInterpolationIndices;
        __private int4 yInterpolationIndices;
        __private int4 combinedInterpolationIndices;
        __private FPTFOUR xWeights;
        __private FPTFOUR yWeights;
        __private int p;
        __private int q;
        __private FPT s;
        __private FPT diff;
        __private FPT Theta;
        
        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            lmask[nIndex] += One;
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

            coord.x -= (coord.x >= Zero) ? (FPT)((int)trunc(coord.x)) : (FPT)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with floor
            coord.y -= (coord.y >= Zero) ? (FPT)((int)trunc(coord.y)) : (FPT)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with floor

            //Calculate the weights for interpolation
            s = One - coord.x;
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
            Theta = yGradient[i] * (FPT)column - xGradient[i] * (FPT)row;
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
        // ensure reads are not out of bounds
        if(i + blockSize < sourcewidth * sourceheight)
        {
            __private int lIdx = i + blockSize;
            column = lIdx % sourcewidth;
            row = (lIdx - column)/sourcewidth;
            coord = (FPTTWO)(offsetx, offsety) + ((FPT)column) * xvec + ((FPT)row) * yvec;            
            Msk = (int2)((int)round(coord.x), (int)round(coord.y));
            if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
            {
                lmask[nIndex] += One;
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
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division it doesn't yield q
                }
                xInterpolationIndices.x = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
                //loop iteration 1
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetWidth)
                {
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division it doesn't yield q
                }
                xInterpolationIndices.y = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
                //loop iteration 2
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetWidth)
                {
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division it doesn't yield q
                }
                xInterpolationIndices.z = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
                //loop iteration 3
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetWidth)
                {
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division it doesn't yield q
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

                coord.x -= (coord.x >= Zero) ? (FPT)((int)trunc(coord.x)) : (FPT)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with floor
                coord.y -= (coord.y >= Zero) ? (FPT)((int)trunc(coord.y)) : (FPT)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with floor

                //Calculate the weights for interpolation
                s = One - coord.x;
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
                Theta = yGradient[lIdx] * (FPT)column - xGradient[lIdx] * (FPT)row;
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


__kernel void translationErrorWithGradAndHessBrent(const __global FPT *source,
const __global FPT *target,
const __global FPT *xGradient,
const __global FPT *yGradient,
__global FPT *grad0,
__global FPT *grad1,
__global FPT *hessian00,
__global FPT *hessian01,
__global FPT *hessian11,
__global FPT *diffout,
__global FPT *mask,
__local volatile FPT *lgrad0,
__local volatile FPT *lgrad1,
__local volatile FPT *lhessian00,
__local volatile FPT *lhessian01,
__local volatile FPT *lhessian11,
__local volatile FPT *ldiffout,
__local volatile FPT *lmask,
const int sourcewidth,
const int sourceheight,
const int targetwidth,
const int targetheight,
const FPT offsetx,
const FPT offsety,
const int doubleTargetWidth,
const int doubleTargetHeight)
{
    //Brent's theorem optimized version to reduce the following sum reduction to blockSize
    __private unsigned int nIndex = get_local_id(0);/*ID within a workgroup*/
    __private unsigned int blockSize = get_local_size(0); //this is the stride
    __private unsigned int i = get_group_id(0)*(blockSize*2) + nIndex;
    __private unsigned int gridSize = blockSize*2*get_num_groups(0);
    
    //prepare local buffers
    lgrad0[nIndex] = Zero;
    lgrad1[nIndex] = Zero;
    lhessian00[nIndex] = Zero;
    lhessian01[nIndex] = Zero;
    lhessian11[nIndex] = Zero;
    ldiffout[nIndex] = Zero;
    lmask[nIndex] = Zero;

    while(i < sourcewidth * sourceheight)
    {
        __private int column = i % sourcewidth;
        __private int row = (i - column)/sourcewidth;
        __private FPTTWO coord = (FPTTWO)(offsetx + ((FPT)column), offsety + ((FPT)row));
        
        __private int4 xInterpolationIndices;
        __private int4 yInterpolationIndices;
        __private int4 combinedInterpolationIndices;
        __private FPTFOUR xWeights;
        __private FPTFOUR yWeights;
        __private int p;
        __private int q;
        __private FPT s;
        __private FPT diff;
        
        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            lmask[nIndex] += One;
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

            coord.x -= (coord.x >= Zero) ? (FPT)((int)trunc(coord.x)) : (FPT)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with floor
            coord.y -= (coord.y >= Zero) ? (FPT)((int)trunc(coord.y)) : (FPT)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with floor

            //Calculate the weights for interpolation
            s = One - coord.x;
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
        // ensure reads are not out of bounds
        if(i + blockSize < sourcewidth * sourceheight)
        {
            __private int lIdx = i + blockSize;
            column = lIdx % sourcewidth;
            row = (lIdx - column)/sourcewidth;
            coord = (FPTTWO)(offsetx + ((FPT)column), offsety + ((FPT)row));
            Msk = (int2)((int)round(coord.x), (int)round(coord.y));
            if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
            {
                lmask[nIndex] += One;
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
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division it doesn't yield q
                }
                xInterpolationIndices.x = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
                //loop iteration 1
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetWidth)
                {
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division it doesn't yield q
                }
                xInterpolationIndices.y = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
                //loop iteration 2
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetWidth)
                {
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division it doesn't yield q
                }
                xInterpolationIndices.z = q >= targetwidth ? (doubleTargetWidth - 1 - q) : q;
                //loop iteration 3
                p--;
                q = abs(p) - rotate(p&(int)0x80000000,(int)1);
                if(q >= doubleTargetWidth)
                {
                    q -= (doubleTargetWidth) * (q / (doubleTargetWidth)); //Warning: this is an integer division it doesn't yield q
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

                coord.x -= (coord.x >= Zero) ? (FPT)((int)trunc(coord.x)) : (FPT)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with floor
                coord.y -= (coord.y >= Zero) ? (FPT)((int)trunc(coord.y)) : (FPT)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with floor

                //Calculate the weights for interpolation
                s = One - coord.x;
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


__kernel void transformImageWithBsplineInterpolation(const __global FPT *source ,__global FPT *target, const int sourcewidth, const int sourceheight, const FPT offsetx, const FPT offsety, const FPT angle)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private FPTTWO xvec = (FPTTWO)(cos(angle),-sin(angle));//warning: this is not the x vector but it is the vector added in the x direction
        __private FPTTWO yvec = (FPTTWO)(-xvec.y,xvec.x);//warning: this is not the y vector but it is the vector added in the y direction
        __private FPTTWO coord = (FPTTWO)(offsetx, offsety) + ((FPT)column) * xvec + ((FPT)row) * yvec;

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
                q -= (2*sourcewidth) * (q / (2*sourcewidth)); //Warning: this is an integer division it doesn't yield q
            }
            xInterpolationIndices.x = q >= sourcewidth ? (2*sourcewidth - 1 - q) : q;
            //loop iteration 1
            p--;
            q = (p < 0) ? (-1 - p) : p;
            if(q >= 2*sourcewidth)
            {
                q -= (2*sourcewidth) * (q / (2*sourcewidth)); //Warning: this is an integer division it doesn't yield q
            }
            xInterpolationIndices.y = q >= sourcewidth ? (2*sourcewidth - 1 - q) : q;
            //loop iteration 2
            p--;
            q = (p < 0) ? (-1 - p) : p;
            if(q >= 2*sourcewidth)
            {
                q -= (2*sourcewidth) * (q / (2*sourcewidth)); //Warning: this is an integer division it doesn't yield q
            }
            xInterpolationIndices.z = q >= sourcewidth ? (2*sourcewidth - 1 - q) : q;
            //loop iteration 3
            p--;
            q = (p < 0) ? (-1 - p) : p;
            if(q >= 2*sourcewidth)
            {
                q -= (2*sourcewidth) * (q / (2*sourcewidth)); //Warning: this is an integer division it doesn't yield q
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

            coord.x -= (coord.x >= Zero) ? (FPT)((int)trunc(coord.x)) : (FPT)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with floor
            coord.y -= (coord.y >= Zero) ? (FPT)((int)trunc(coord.y)) : (FPT)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with floor

            //Calculate the weights for interpolation
            __private FPTFOUR xWeights;
            __private FPT s = One - coord.x;
            xWeights.w = pown(s,3) / Lambda;
            s = coord.x * coord.x;
            xWeights.z = Two / Three - h * s * (Two - coord.x);
            xWeights.x = s * coord.x / Lambda;
            xWeights.y = One - xWeights.x - xWeights.z - xWeights.w;
            __private FPTFOUR yWeights;
            s = One - coord.y;
            yWeights.w = pown(s,3) / Lambda;
            s = coord.y * coord.y;
            yWeights.z = Two / Three - h * s * (Two - coord.y);
            yWeights.x = s * coord.y / Lambda;
            yWeights.y = One - yWeights.x - yWeights.z - yWeights.w;

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
            target[nIndex] = Zero;
        } 
    }   
}

__kernel void translationTransformImageWithBsplineInterpolation(const __global FPT *source ,__global FPT *target, const int sourcewidth, const int sourceheight, const FPT offsetx, const FPT offsety)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private FPTTWO coord = (FPTTWO)(offsetx + ((FPT)column), offsety + ((FPT)row));

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
                q -= (2*sourcewidth) * (q / (2*sourcewidth)); //Warning: this is an integer division it doesn't yield q
            }
            xInterpolationIndices.x = q >= sourcewidth ? (2*sourcewidth - 1 - q) : q;
            //loop iteration 1
            p--;
            q = (p < 0) ? (-1 - p) : p;
            if(q >= 2*sourcewidth)
            {
                q -= (2*sourcewidth) * (q / (2*sourcewidth)); //Warning: this is an integer division it doesn't yield q
            }
            xInterpolationIndices.y = q >= sourcewidth ? (2*sourcewidth - 1 - q) : q;
            //loop iteration 2
            p--;
            q = (p < 0) ? (-1 - p) : p;
            if(q >= 2*sourcewidth)
            {
                q -= (2*sourcewidth) * (q / (2*sourcewidth)); //Warning: this is an integer division it doesn't yield q
            }
            xInterpolationIndices.z = q >= sourcewidth ? (2*sourcewidth - 1 - q) : q;
            //loop iteration 3
            p--;
            q = (p < 0) ? (-1 - p) : p;
            if(q >= 2*sourcewidth)
            {
                q -= (2*sourcewidth) * (q / (2*sourcewidth)); //Warning: this is an integer division it doesn't yield q
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

            coord.x -= (coord.x >= Zero) ? (FPT)((int)trunc(coord.x)) : (FPT)(((int)trunc(coord.x)) - 1);//get the residual should also be possible with floor
            coord.y -= (coord.y >= Zero) ? (FPT)((int)trunc(coord.y)) : (FPT)(((int)trunc(coord.y)) - 1);//get the residual should also be possible with floor

            //Calculate the weights for interpolation
            __private FPTFOUR xWeights;
            __private FPT s = One - coord.x;
            xWeights.w = pown(s,3) / Lambda;
            s = coord.x * coord.x;
            xWeights.z = Two / Three - h * s * (Two - coord.x);
            xWeights.x = s * coord.x / Lambda;
            xWeights.y = One - xWeights.x - xWeights.z - xWeights.w;
            __private FPTFOUR yWeights;
            s = One - coord.y;
            yWeights.w = pown(s,3) / Lambda;
            s = coord.y * coord.y;
            yWeights.z = Two / Three - h * s * (Two - coord.y);
            yWeights.x = s * coord.y / Lambda;
            yWeights.y = One - yWeights.x - yWeights.z - yWeights.w;

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
            target[nIndex] = Zero;
        } 
    }   
}