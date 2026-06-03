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
Commands to build SPIRV binaries for the kernels:

clang -D USE_DOUBLE -D AFFINE -target spirv64 -cl-std=CL1.2 -Xclang -fdeclare-opencl-builtins -c -O3 UniformBSplineTransform.cl -o UniformBSplineTransform_affine_double_64.spv
clang -D USE_DOUBLE -D SCALEDROTATION -target spirv64 -cl-std=CL1.2 -Xclang -fdeclare-opencl-builtins -c -O3 UniformBSplineTransform.cl -o UniformBSplineTransform_scaledrotation_double_64.spv
clang -D USE_DOUBLE -D RIGIDBODY -target spirv64 -cl-std=CL1.2 -Xclang -fdeclare-opencl-builtins -c -O3 UniformBSplineTransform.cl -o UniformBSplineTransform_rigidbody_double_64.spv
clang -D USE_DOUBLE -D TRANSLATION -target spirv64 -cl-std=CL1.2 -Xclang -fdeclare-opencl-builtins -c -O3 UniformBSplineTransform.cl -o UniformBSplineTransform_translation_double_64.spv
clang -D TRANSLATION -target spirv64 -cl-std=CL1.2 -Xclang -fdeclare-opencl-builtins -c -O3 UniformBSplineTransform.cl -o UniformBSplineTransform_translation_float_64.spv
clang -D RIGIDBODY -target spirv64 -cl-std=CL1.2 -Xclang -fdeclare-opencl-builtins -c -O3 UniformBSplineTransform.cl -o UniformBSplineTransform_rigidbody_float_64.spv
clang -D SCALEDROTATION -target spirv64 -cl-std=CL1.2 -Xclang -fdeclare-opencl-builtins -c -O3 UniformBSplineTransform.cl -o UniformBSplineTransform_scaledrotation_float_64.spv
clang -D AFFINE -target spirv64 -cl-std=CL1.2 -Xclang -fdeclare-opencl-builtins -c -O3 UniformBSplineTransform.cl -o UniformBSplineTransform_affine_float_64.spv
clang -D HYBRID -D AFFINE -target spirv64 -cl-std=CL1.2 -Xclang -fdeclare-opencl-builtins -c -O3 UniformBSplineTransform.cl -o UniformBSplineTransform_hybrid_affine_float_64.spv
clang -D HYBRID -D SCALEDROTATION -target spirv64 -cl-std=CL1.2 -Xclang -fdeclare-opencl-builtins -c -O3 UniformBSplineTransform.cl -o UniformBSplineTransform_hybrid_scaledrotation_float_64.spv
clang -D HYBRID -D RIGIDBODY -target spirv64 -cl-std=CL1.2 -Xclang -fdeclare-opencl-builtins -c -O3 UniformBSplineTransform.cl -o UniformBSplineTransform_hybrid_rigidbody_float_64.spv
clang -D HYBRID -D TRANSLATION -target spirv64 -cl-std=CL1.2 -Xclang -fdeclare-opencl-builtins -c -O3 UniformBSplineTransform.cl -o UniformBSplineTransform_hybrid_translation_float_64.spv
clang -D HYBRID -D TRANSLATION -D USE_DOUBLE -target spirv64 -cl-std=CL1.2 -Xclang -fdeclare-opencl-builtins -c -O3 UniformBSplineTransform.cl -o UniformBSplineTransform_hybrid_translation_double_64.spv
clang -D HYBRID -D RIGIDBODY -D USE_DOUBLE -target spirv64 -cl-std=CL1.2 -Xclang -fdeclare-opencl-builtins -c -O3 UniformBSplineTransform.cl -o UniformBSplineTransform_hybrid_rigidbody_double_64.spv
clang -D HYBRID -D SCALEDROTATION -D USE_DOUBLE -target spirv64 -cl-std=CL1.2 -Xclang -fdeclare-opencl-builtins -c -O3 UniformBSplineTransform.cl -o UniformBSplineTransform_hybrid_scaledrotation_double_64.spv
clang -D HYBRID -D AFFINE -D USE_DOUBLE -target spirv64 -cl-std=CL1.2 -Xclang -fdeclare-opencl-builtins -c -O3 UniformBSplineTransform.cl -o UniformBSplineTransform_hybrid_affine_double_64.spv
*/

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
__constant const int Horizon = 12;
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

#if defined(HYBRID) && !defined(USE_DOUBLE)
//low precision variant with horizon for float precision (faster) from 10.1093/comjnl/bxq086
__kernel void fCubicBSplinePrefilter2DXlp(__global float *image, const int width, const int height)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the row!!!
    if(nIndex < height)
    {
        __global float *prow = image + (nIndex * width);
        //causal initialization
        __private float zk = Pole;
        __private float Sum = prow[0];
        for(int k = 0; k < Horizon; k++)
        {
            //Sum += zk * prow[k];
            Sum = fma(zk, prow[k], Sum);
            zk *= Pole;
        }
        prow[0] = Sum;
        //Causal recursion
        for(int k = 1; k < width; k++)
        {
            //prow[k] += Pole *  prow[k-1];
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
#else
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
#endif

#if defined(HYBRID) && !defined(USE_DOUBLE)
//low precision variant with horizon for float precision (faster) from 10.1093/comjnl/bxq086
__kernel void fCubicBSplinePrefilter2DYlp(__global float *image, const int width, const int height)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the column!!!
    if(nIndex < width)
    {
        __global float *pcol = image + nIndex;
    
        //causal initialization
        __private float zk = Pole;
        __private float Sum = pcol[0];
        for(int k = 0; k < Horizon; k++)
        {
            //Sum += zk * pcol[k * width];
            Sum = fma(zk, pcol[k * width], Sum);
            zk *= Pole;
        }
        pcol[0] = Sum;

        //Causal recursion
        for(int k = 1; k < height; k++)
        {
            //pcol[k * width] += Pole *  pcol[(k-1)*width];
            pcol[k * width] = fma(Pole,  pcol[(k-1)*width], pcol[k * width]);
        }

        //anticausal initialization
        pcol[(height - 1)*width] = (Pole * pcol[(height - 1)*width] / (Pole - One));

        //Anticausal recursion
        for(int k = height - 2; 0 <= k; k--)
        {
            pcol[k*width] = (Pole * (pcol[(k+1)*width] -  pcol[k*width]));
        }
    }    
}
#else
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
#endif

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

#if 0
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
        __private float zk = Z0;
        __private float Sum = prow[0];
        for(int k = 0; k < Horizon; k++)
        {
            //Sum += zk * prow[k];
            Sum = fma(zk, prow[k], Sum);
            zk *= Z0;
        }
        prow[0] = Sum;

    
        //Causal recursion
        for(int k = 1; k < width; k++)
        {
            //prow[k] += Z0 *  prow[k-1];
            prow[k] = fma(Z0, prow[k-1], prow[k]);
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
        zk = Z1;
        Sum = prow[0];
        for(int k = 0; k < Horizon; k++)
        {
            //Sum += zk * prow[k];
            Sum = fma(zk, prow[k], Sum);
            zk *= Z1;
        }
        prow[0] = Sum;

        //Causal recursion
        for(int k = 1; k < width; k++)
        {
            //prow[k] += Z1 *  prow[k-1];
            prow[k] = fma(Z1,  prow[k-1], prow[k]);
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
        zk = Z2;
        Sum = prow[0];
        for(int k = 0; k < Horizon; k++)
        {
            //Sum += zk * prow[k];
            Sum = fma(zk, prow[k], Sum);
            zk *= Z2;
        }
        prow[0] = Sum;
    
        //Causal recursion
        for(int k = 1; k < width; k++)
        {
            //prow[k] += Z2 * prow[k-1];
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
#else
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
#endif

#if 0
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
        __private float zk = Z0;
        __private float Sum = prow[0];
        for(int k = 0; k < Horizon; k++)
        {
            //Sum += zk * prow[k * width];
            Sum = fma(zk, prow[k * width], Sum);
            zk *= Z0;
        }
        prow[0] = Sum;

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
        zk = Z1;
        Sum = prow[0];
        for(int k = 0; k < Horizon; k++)
        {
            //Sum += zk * prow[k * width];
            Sum = fma(zk, prow[k * width], Sum);
            zk *= Z1;
        }
        prow[0] = Sum;
    
        //Causal recursion
        for(int k = 1; k < height; k++)
        {
            //prow[k*width] += Z1 *  prow[(k-1)*width];
            prow[k*width] = fma(Z1,  prow[(k-1)*width], prow[k*width]);
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
        zk = Z2;
        Sum = prow[0];
        for(int k = 0; k < Horizon; k++)
        {
            //Sum += zk * prow[k * width];
            Sum = fma(zk, prow[k * width], Sum);
            zk *= Z2;
        }
        prow[0] = Sum;
    
        //Causal recursion
        for(int k = 1; k < height; k++)
        {
            //prow[k*width] += Z2 *  prow[(k-1)*width];
            prow[k*width] = fma(Z2,  prow[(k-1)*width], prow[k*width]);
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
#else
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
#endif

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

#ifdef TRANSLATION
//calculate the square error
__kernel void translationError(const __global FPT *source ,const __global FPT *target, __global FPT *diffout, __global FPT *mask, const int sourcewidth, const int sourceheight, const int targetwidth, const int targetheight, const int doubletargetwidth, const int doubletargetheight, const FPT offsetx, const FPT offsety)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private FPTTWO coord = (FPTTWO)(offsetx + ((FPT)column), offsety + ((FPT)row));

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

__kernel void translationErrorWithGradAndHess(const __global FPT *source ,const __global FPT *target,const __global FPT *xGradient,const __global FPT *yGradient,__global FPT *grad0,__global FPT *grad1,__global FPT *hessian00,__global FPT *hessian01,__global FPT *hessian11, __global FPT *diffout, __global FPT *mask, const int sourcewidth, const int sourceheight, const int targetwidth, const int targetheight, const int doubletargetwidth, const int doubletargetheight, const FPT offsetx, const FPT offsety)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private FPTTWO coord = (FPTTWO)(offsetx + ((FPT)column), offsety + ((FPT)row));

        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        __private int4 xInterpolationIndxs;
        __private int4 yInterpolationIndxs;
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            mask[nIndex] = One;
            xInterpolationIndxs = calculatexInterpolationIndxs(coord.x, doubletargetwidth, targetwidth);
            yInterpolationIndxs = calculateyInterpolationIndxs(coord.y, doubletargetheight, targetheight, targetwidth);
            __private FPT s = interpolate(coord, xInterpolationIndxs, yInterpolationIndxs, target);
            __private FPT diff = source[nIndex] - s;
            //diffout[nIndex] = pown(diff,2);
            //grad0[nIndex] = diff * xGradient[nIndex];
            //grad1[nIndex] = diff * yGradient[nIndex];
            __private FPTTHREE tmp = (FPTTHREE)(diff, xGradient[nIndex], yGradient[nIndex]) * diff;
            diffout[nIndex] = tmp.x;
            grad0[nIndex] = tmp.y;
            grad1[nIndex] = tmp.z;
            //hessian00[nIndex] = pown(xGradient[nIndex],2);
            //hessian01[nIndex] = xGradient[nIndex] * yGradient[nIndex];
            //hessian11[nIndex] = pown(yGradient[nIndex],2);
            tmp = ((FPTTHREE)(xGradient[nIndex], xGradient[nIndex], yGradient[nIndex])) * ((FPTTHREE)(xGradient[nIndex], yGradient[nIndex], yGradient[nIndex])); 
            hessian00[nIndex] = tmp.x;
            hessian01[nIndex] = tmp.y;
            hessian11[nIndex] = tmp.z;
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

    __private const int totalPixels = sourcewidth * sourceheight;
    while(i < totalPixels)
    {
        __private int column = i % sourcewidth;
        __private int row = (i - column)/sourcewidth;
        __private FPTTWO coord = (FPTTWO)(offsetx, offsety) + (FPTTWO)(((FPT)column), ((FPT)row));
        
        __private int4 xInterpolationIndxs;
        __private int4 yInterpolationIndxs;
        __private int4 combinedInterpolationIndices;
        __private FPT s;
        __private FPT diff;
        
        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            lmask[nIndex] += One;
            xInterpolationIndxs = calculatexInterpolationIndxs(coord.x, doubleTargetWidth, targetwidth);
            yInterpolationIndxs = calculateyInterpolationIndxs(coord.y, doubleTargetHeight, targetheight, targetwidth);
            s = interpolate(coord, xInterpolationIndxs, yInterpolationIndxs, target);
            diff = source[i] - s;
            //ldiffout[nIndex] += pown(diff,2);
            //lgrad0[nIndex] += diff * xGradient[i];
            //lgrad1[nIndex] += diff * yGradient[i];
            __private FPTTHREE tmp3 = fma((FPTTHREE)(diff, xGradient[i], yGradient[i]), (FPTTHREE)diff, (FPTTHREE)(ldiffout[nIndex], lgrad0[nIndex], lgrad1[nIndex]));
            ldiffout[nIndex] = tmp3.x;
            lgrad0[nIndex] = tmp3.y;
            lgrad1[nIndex] = tmp3.z;
            //lhessian00[nIndex] += pown(xGradient[i],2);
            //lhessian01[nIndex] += xGradient[i] * yGradient[i];
            //lhessian11[nIndex] += pown(yGradient[i],2);
            tmp3 = fma((FPTTHREE)(xGradient[i], xGradient[i], yGradient[i]), (FPTTHREE)(xGradient[i], yGradient[i], yGradient[i]), (FPTTHREE)(lhessian00[nIndex], lhessian01[nIndex], lhessian11[nIndex]));
            lhessian00[nIndex] = tmp3.x;
            lhessian01[nIndex] = tmp3.y;
            lhessian11[nIndex] = tmp3.z;
        }
        // ensure reads are not out of bounds
        if(i + blockSize < totalPixels)
        {
            __private int lIdx = i + blockSize;
            column = lIdx % sourcewidth;
            row = (lIdx - column)/sourcewidth;
            coord = (FPTTWO)(offsetx, offsety) + (FPTTWO)(((FPT)column), ((FPT)row));
            Msk = (int2)((int)round(coord.x), (int)round(coord.y));
            if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
            {
                lmask[nIndex] += One;
                xInterpolationIndxs = calculatexInterpolationIndxs(coord.x, doubleTargetWidth, targetwidth);
                yInterpolationIndxs = calculateyInterpolationIndxs(coord.y, doubleTargetHeight, targetheight, targetwidth);
                s = interpolate(coord, xInterpolationIndxs, yInterpolationIndxs, target);
                diff = source[lIdx] - s;
                //ldiffout[nIndex] += pown(diff,2);
                //lgrad0[nIndex] += diff * xGradient[lIdx];
                //lgrad1[nIndex] += diff * yGradient[lIdx];
                __private FPTTHREE tmp3 = fma((FPTTHREE)(diff, xGradient[lIdx], yGradient[lIdx]), (FPTTHREE)diff, (FPTTHREE)(ldiffout[nIndex], lgrad0[nIndex], lgrad1[nIndex]));
                ldiffout[nIndex] = tmp3.x;
                lgrad0[nIndex] = tmp3.y;
                lgrad1[nIndex] = tmp3.z;
                //lhessian00[nIndex] += pown(xGradient[lIdx],2);
                //lhessian01[nIndex] += xGradient[lIdx] * yGradient[lIdx];
                //lhessian11[nIndex] += pown(yGradient[lIdx],2);
                tmp3 = fma((FPTTHREE)(xGradient[lIdx], xGradient[lIdx], yGradient[lIdx]), (FPTTHREE)(xGradient[lIdx], yGradient[lIdx], yGradient[lIdx]), (FPTTHREE)(lhessian00[nIndex], lhessian01[nIndex], lhessian11[nIndex]));
                lhessian00[nIndex] = tmp3.x;
                lhessian01[nIndex] = tmp3.y;
                lhessian11[nIndex] = tmp3.z;
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

__kernel void translationTransformImageWithBsplineInterpolation(const __global FPT *source ,__global FPT *target, const int sourcewidth, const int sourceheight, const int doubleSourceWidth, const int doubleSourceHeight, const FPT offsetx, const FPT offsety)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private FPTTWO coord = (FPTTWO)(offsetx, offsety) + (FPTTWO)(((FPT)column), ((FPT)row));

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

__kernel void resizingTranslationTransformImageWithBsplineInterpolation(const __global FPT *source ,__global FPT *target, const int sourcewidth, const int sourceheight, const int doubleSourceWidth, const int doubleSourceHeight, const int targetwidth, const int targetheight, const FPT offsetx, const FPT offsety)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !TARGET! pixel
    if(nIndex < targetwidth * targetheight)
    {
        __private int column = nIndex % targetwidth;
        __private int row = (nIndex - column)/targetwidth;

        __private FPTTWO coord = (FPTTWO)(offsetx, offsety) + (FPTTWO)(((FPT)column), ((FPT)row));

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
#endif

#ifdef RIGIDBODY
//calculate the square error
__kernel void rigidBodyError(const __global FPT *source ,const __global FPT *target, __global FPT *diffout, __global FPT *mask, const int sourcewidth, const int sourceheight, const int targetwidth, const int targetheight, const int doubletargetwidth, const int doubletargetheight, const FPT offsetx, const FPT offsety, const FPT cosangle, const FPT negsinangle)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private FPTTWO xvec = (FPTTWO)(cosangle,negsinangle);//warning: this is not the x vector but it is the vector added in the x direction
        __private FPTTWO yvec = (FPTTWO)(-xvec.y,xvec.x);//warning: this is not the y vector but it is the vector added in the y direction
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

__kernel void rigidBodyErrorWithGradAndHess(const __global FPT *source ,const __global FPT *target,const __global FPT *xGradient,const __global FPT *yGradient,__global FPT *grad0,__global FPT *grad1,__global FPT *grad2,__global FPT *hessian00,__global FPT *hessian01,__global FPT *hessian02,__global FPT *hessian11,__global FPT *hessian12,__global FPT *hessian22, __global FPT *diffout, __global FPT *mask, const int sourcewidth, const int sourceheight, const int targetwidth, const int targetheight, const int doubletargetwidth, const int doubletargetheight, const FPT offsetx, const FPT offsety, const FPT cosangle, const FPT negsinangle)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private FPTTWO xvec = (FPTTWO)(cosangle,negsinangle);//warning: this is not the x vector but it is the vector added in the x direction
        __private FPTTWO yvec = (FPTTWO)(-xvec.y,xvec.x);//warning: this is not the y vector but it is the vector added in the y direction
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
            __private FPT diff = source[nIndex] - s;
            //__private FPT Theta = yGradient[nIndex] * (FPT)column - xGradient[nIndex] * (FPT)row;
            __private FPT Theta = dot((FPTTWO)(yGradient[nIndex], -xGradient[nIndex]), (FPTTWO)((FPT)column, (FPT)row));
            //diffout[nIndex] = pown(diff,2);
            //grad0[nIndex] = diff * Theta;
            //grad1[nIndex] = diff * xGradient[nIndex];
            //grad2[nIndex] = diff * yGradient[nIndex];
            __private FPTFOUR tmp4 = (FPTFOUR)(diff, Theta, xGradient[nIndex], yGradient[nIndex]) * diff; //Slightly less accurate
            diffout[nIndex] = tmp4.x;
            grad0[nIndex] = tmp4.y;
            grad1[nIndex] = tmp4.z;
            grad2[nIndex] = tmp4.w;
            //hessian00[nIndex] = pown(Theta,2); //this is more accurate
            //hessian01[nIndex] = Theta * xGradient[nIndex];
            //hessian02[nIndex] = Theta * yGradient[nIndex];
            __private FPTTHREE tmp = (FPTTHREE)(Theta, xGradient[nIndex], yGradient[nIndex]) * Theta; //Slightly less accurate
            hessian00[nIndex] = tmp.x;
            hessian01[nIndex] = tmp.y;
            hessian02[nIndex] = tmp.z;
            //hessian11[nIndex] = pown(xGradient[nIndex],2);
            //hessian12[nIndex] = xGradient[nIndex] * yGradient[nIndex];
            //hessian22[nIndex] = pown(yGradient[nIndex],2);
            tmp = ((FPTTHREE)(xGradient[nIndex], xGradient[nIndex], yGradient[nIndex])) * ((FPTTHREE)(xGradient[nIndex], yGradient[nIndex], yGradient[nIndex])); //Slightly less accurate
            hessian11[nIndex] = tmp.x;
            hessian12[nIndex] = tmp.y;
            hessian22[nIndex] = tmp.z;
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

__kernel void sumInLocalMemoryCombined(__global FPT *gdata0, //maskBuffer
__global FPT *gdata1, //entryImageBuffer
__global FPT *gdata2, //gradient0
__global FPT *gdata3, //gradient1
__global FPT *gdata4, //gradient2
__global FPT *gdata5, //hessian00
__global FPT *gdata6, //hessian01
__global FPT *gdata7, //hessian02
__global FPT *gdata8, //hessian11
__global FPT *gdata9, //hessian12
__global FPT *gdata10, //hessian22
__local volatile FPT* ldata, 
const int size)
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
const FPT cosangle,
const FPT negsinangle,
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
    __private FPTTWO xvec = (FPTTWO)(cosangle,negsinangle);//warning: this is not the x vector but it is the vector added in the x direction
    __private FPTTWO yvec = (FPTTWO)(-xvec.y,xvec.x);//warning: this is not the y vector but it is the vector added in the y direction
    __private const int totalPixels = sourcewidth * sourceheight;
    while(i < totalPixels)
    {
        __private int column = i % sourcewidth;
        __private int row = (i - column)/sourcewidth;
        __private FPTTWO coord = (FPTTWO)(offsetx, offsety) + ((FPT)column) * xvec + ((FPT)row) * yvec;
        
        __private int4 xInterpolationIndxs;
        __private int4 yInterpolationIndxs;
        __private int4 combinedInterpolationIndices;
        __private FPT s;
        __private FPT diff;
        __private FPT Theta;
        
        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            lmask[nIndex] += One;
            xInterpolationIndxs = calculatexInterpolationIndxs(coord.x, doubleTargetWidth, targetwidth);
            yInterpolationIndxs = calculateyInterpolationIndxs(coord.y, doubleTargetHeight, targetheight, targetwidth);            
            s = interpolate(coord, xInterpolationIndxs, yInterpolationIndxs, target);
            diff = source[i] - s;
            //Theta = yGradient[i] * (FPT)column - xGradient[i] * (FPT)row;
            Theta = dot((FPTTWO)(yGradient[i], -xGradient[i]), (FPTTWO)((FPT)column, (FPT)row));
            //ldiffout[nIndex] += pown(diff,2);
            //lgrad0[nIndex] += diff * Theta;
            //lgrad1[nIndex] += diff * xGradient[i];
            //lgrad2[nIndex] += diff * yGradient[i];
            __private FPTFOUR tmp4 = fma((FPTFOUR)(diff, Theta, xGradient[i], yGradient[i]), (FPTFOUR)diff, (FPTFOUR)(ldiffout[nIndex], lgrad0[nIndex], lgrad1[nIndex], lgrad2[nIndex]));
            ldiffout[nIndex] = tmp4.x;
            lgrad0[nIndex] = tmp4.y;
            lgrad1[nIndex] = tmp4.z;
            lgrad2[nIndex] = tmp4.w;
            //lhessian00[nIndex] += pown(Theta,2);
            //lhessian01[nIndex] += Theta * xGradient[i];
            //lhessian02[nIndex] += Theta * yGradient[i];
            __private FPTTHREE tmp = fma((FPTTHREE)(Theta, xGradient[i], yGradient[i]), (FPTTHREE)Theta, (FPTTHREE)(lhessian00[nIndex], lhessian01[nIndex], lhessian02[nIndex])); //Slightly less accurate
            lhessian00[nIndex] = tmp.x;
            lhessian01[nIndex] = tmp.y;
            lhessian02[nIndex] = tmp.z;
            //lhessian11[nIndex] += pown(xGradient[i],2);
            //lhessian12[nIndex] += xGradient[i] * yGradient[i];
            //lhessian22[nIndex] += pown(yGradient[i],2);
            tmp = fma((FPTTHREE)(xGradient[i], xGradient[i], yGradient[i]), (FPTTHREE)(xGradient[i], yGradient[i], yGradient[i]), (FPTTHREE)(lhessian11[nIndex], lhessian12[nIndex], lhessian22[nIndex]));
            lhessian11[nIndex] = tmp.x;
            lhessian12[nIndex] = tmp.y;
            lhessian22[nIndex] = tmp.z;
        }
        // ensure reads are not out of bounds
        if(i + blockSize < totalPixels)
        {
            __private int lIdx = i + blockSize;
            column = lIdx % sourcewidth;
            row = (lIdx - column)/sourcewidth;
            coord = (FPTTWO)(offsetx, offsety) + ((FPT)column) * xvec + ((FPT)row) * yvec;            
            Msk = (int2)((int)round(coord.x), (int)round(coord.y));
            if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
            {
                lmask[nIndex] += One;
                xInterpolationIndxs = calculatexInterpolationIndxs(coord.x, doubleTargetWidth, targetwidth);
                yInterpolationIndxs = calculateyInterpolationIndxs(coord.y, doubleTargetHeight, targetheight, targetwidth); 
                s = interpolate(coord, xInterpolationIndxs, yInterpolationIndxs, target);
                diff = source[lIdx] - s;
                //Theta = yGradient[lIdx] * (FPT)column - xGradient[lIdx] * (FPT)row;
                Theta = dot((FPTTWO)(yGradient[lIdx], -xGradient[lIdx]), (FPTTWO)((FPT)column, (FPT)row));
                //ldiffout[nIndex] += pown(diff,2);
                //lgrad0[nIndex] += diff * Theta;
                //lgrad1[nIndex] += diff * xGradient[lIdx];
                //lgrad2[nIndex] += diff * yGradient[lIdx];
                __private FPTFOUR tmp4 = fma((FPTFOUR)(diff, Theta, xGradient[lIdx], yGradient[lIdx]), (FPTFOUR)diff, (FPTFOUR)(ldiffout[nIndex], lgrad0[nIndex], lgrad1[nIndex], lgrad2[nIndex]));
                ldiffout[nIndex] = tmp4.x;
                lgrad0[nIndex] = tmp4.y;
                lgrad1[nIndex] = tmp4.z;
                lgrad2[nIndex] = tmp4.w;
                //lhessian00[nIndex] += pown(Theta,2);
                //lhessian01[nIndex] += Theta * xGradient[lIdx];
                //lhessian02[nIndex] += Theta * yGradient[lIdx];
                __private FPTTHREE tmp = fma((FPTTHREE)(Theta, xGradient[lIdx], yGradient[lIdx]), (FPTTHREE)Theta, (FPTTHREE)(lhessian00[nIndex], lhessian01[nIndex], lhessian02[nIndex])); //Slightly less accurate
                lhessian00[nIndex] = tmp.x;
                lhessian01[nIndex] = tmp.y;
                lhessian02[nIndex] = tmp.z;
                //lhessian11[nIndex] += pown(xGradient[lIdx],2);
                //lhessian12[nIndex] += xGradient[lIdx] * yGradient[lIdx];
                //lhessian22[nIndex] += pown(yGradient[lIdx],2);
                tmp = fma((FPTTHREE)(xGradient[lIdx], xGradient[lIdx], yGradient[lIdx]), (FPTTHREE)(xGradient[lIdx], yGradient[lIdx], yGradient[lIdx]), (FPTTHREE)(lhessian11[nIndex], lhessian12[nIndex], lhessian22[nIndex]));
                lhessian11[nIndex] = tmp.x;
                lhessian12[nIndex] = tmp.y;
                lhessian22[nIndex] = tmp.z;
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

__kernel void transformImageWithBsplineInterpolation(const __global FPT *source ,__global FPT *target, const int sourcewidth, const int sourceheight, const int doubleSourceWidth, const int doubleSourceHeight, const FPT offsetx, const FPT offsety, const FPT cosangle, const FPT negsinangle)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private FPTTWO xvec = (FPTTWO)(cosangle, negsinangle);//warning: this is not the x vector but it is the vector added in the x direction
        __private FPTTWO yvec = (FPTTWO)(-xvec.y, xvec.x);//warning: this is not the y vector but it is the vector added in the y direction
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

__kernel void resizingTransformImageWithBsplineInterpolation(const __global FPT *source ,__global FPT *target, const int sourcewidth, const int sourceheight, const int doubleSourceWidth, const int doubleSourceHeight, const int targetwidth, const int targetheight, const FPT offsetx, const FPT offsety, const FPT cosangle, const FPT negsinangle)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !TARGET! pixel
    if(nIndex < targetwidth * targetheight)
    {
        __private int column = nIndex % targetwidth;
        __private int row = (nIndex - column)/targetwidth;

        __private FPTTWO xvec = (FPTTWO)(cosangle, negsinangle);//warning: this is not the x vector but it is the vector added in the x direction
        __private FPTTWO yvec = (FPTTWO)(-xvec.y, xvec.x);//warning: this is not the y vector but it is the vector added in the y direction
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
#endif

#ifdef SCALEDROTATION
//calculate the square error
__kernel void scaledRotationError(const __global FPT *source ,const __global FPT *target, __global FPT *diffout, __global FPT *mask, const int sourcewidth, const int sourceheight, const int targetwidth, const int targetheight, const int doubletargetwidth, const int doubletargetheight, const FPT offsetx, const FPT offsety, const FPT cosangle, const FPT negsinangle, const FPT scale)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;
        
        __private FPTTWO xvec = (FPTTWO)(cosangle,negsinangle) * scale;//warning: this is not the x vector but it is the vector added in the x direction
        __private FPTTWO yvec = (FPTTWO)(-xvec.y,xvec.x);//warning: this is not the y vector but it is the vector added in the y direction
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

__kernel void scaledRotationErrorWithGradAndHess(const __global FPT *source,
const __global FPT *target,
const __global FPT *xGradient,
const __global FPT *yGradient,
__global FPT *grad0,
__global FPT *grad1,
__global FPT *grad2,
__global FPT *grad3,
__global FPT *hessian00,
__global FPT *hessian01,
__global FPT *hessian02,
__global FPT *hessian03,
__global FPT *hessian11,
__global FPT *hessian12,
__global FPT *hessian13,
__global FPT *hessian22,
__global FPT *hessian23,
__global FPT *hessian33, 
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
const FPT cosangle, 
const FPT negsinangle, 
const FPT scale)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private FPTTWO xvec = (FPTTWO)(cosangle,negsinangle) * scale;//warning: this is not the x vector but it is the vector added in the x direction
        __private FPTTWO yvec = (FPTTWO)(-xvec.y,xvec.x);//warning: this is not the y vector but it is the vector added in the y direction
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
            __private FPT diff = source[nIndex] - s;
            //__private FPT Theta = yGradient[nIndex] * (FPT)column - xGradient[nIndex] * (FPT)row;
            __private FPT Theta = dot((FPTTWO)(yGradient[nIndex], -xGradient[nIndex]), (FPTTWO)((FPT)column, (FPT)row));
            
            /*
             * Switched to log scale for the scale parameter, which is more stable for optimization. 
             * The original implementation used a linear scale parameter, but this can lead to large 
             * updates and instability when the scale is far from 1. By optimizing in log space, 
             * we ensure that updates are multiplicative and more stable across a wide range of scales.
             */
            //__private FPT j_scale = dot((FPTTWO)(xGradient[nIndex], yGradient[nIndex]), (FPTTWO)((FPT)column, (FPT)row));
            __private FPT j_logScale = dot((FPTTWO)(xGradient[nIndex], yGradient[nIndex]), (FPTTWO)((FPT)column, (FPT)row)) * scale; //chain rule for log scale, this is the derivative of the error with respect to log(scale)
            
            //diffout[nIndex] = pown(diff,2);
            //grad0[nIndex] = diff * j_logScale;
            //grad1[nIndex] = diff * Theta;
            //grad2[nIndex] = diff * xGradient[nIndex];
            //grad3[nIndex] = diff * yGradient[nIndex];
            diffout[nIndex] = diff * diff;
            __private FPTFOUR tmp4 = (FPTFOUR)(j_logScale, Theta, xGradient[nIndex], yGradient[nIndex]) * diff; //Slightly less accurate
            grad0[nIndex] = tmp4.x;
            grad1[nIndex] = tmp4.y;
            grad2[nIndex] = tmp4.z;
            grad3[nIndex] = tmp4.w;
            //hessian00[nIndex] = pown(j_logScale,2); //this is more accurate
            //hessian01[nIndex] = j_logScale * Theta;
            //hessian02[nIndex] = j_logScale * xGradient[nIndex];
            //hessian03[nIndex] = j_logScale * yGradient[nIndex];
            tmp4 = (FPTFOUR)(j_logScale, Theta, xGradient[nIndex], yGradient[nIndex]) * j_logScale; //Slightly less accurate
            hessian00[nIndex] = tmp4.x;
            hessian01[nIndex] = tmp4.y;
            hessian02[nIndex] = tmp4.z;
            hessian03[nIndex] = tmp4.w;
            
            //hessian11[nIndex] = pown(Theta,2); //this is more accurate
            //hessian12[nIndex] = Theta * xGradient[nIndex];
            //hessian13[nIndex] = Theta * yGradient[nIndex];
            __private FPTTHREE tmp = (FPTTHREE)(Theta, xGradient[nIndex], yGradient[nIndex]) * Theta; //Slightly less accurate
            hessian11[nIndex] = tmp.x;
            hessian12[nIndex] = tmp.y;
            hessian13[nIndex] = tmp.z;
            
            //hessian22[nIndex] = pown(xGradient[nIndex],2);
            //hessian23[nIndex] = xGradient[nIndex] * yGradient[nIndex];
            //hessian33[nIndex] = pown(yGradient[nIndex],2);
            tmp = ((FPTTHREE)(xGradient[nIndex], xGradient[nIndex], yGradient[nIndex])) * ((FPTTHREE)(xGradient[nIndex], yGradient[nIndex], yGradient[nIndex])); //Slightly less accurate
            hessian22[nIndex] = tmp.x;
            hessian23[nIndex] = tmp.y;
            hessian33[nIndex] = tmp.z;
        }
        else
        {
            grad0[nIndex] = Zero;
            grad1[nIndex] = Zero;
            grad2[nIndex] = Zero;
            grad3[nIndex] = Zero;
            hessian00[nIndex] = Zero;
            hessian01[nIndex] = Zero;
            hessian02[nIndex] = Zero;
            hessian03[nIndex] = Zero;
            hessian11[nIndex] = Zero;
            hessian12[nIndex] = Zero;
            hessian13[nIndex] = Zero;
            hessian22[nIndex] = Zero;
            hessian23[nIndex] = Zero;
            hessian33[nIndex] = Zero;
            diffout[nIndex] = Zero;
            mask[nIndex] = Zero;
        }
    }
}

__kernel void sumInLocalMemoryScaledRotationCombined(__global FPT *gdata0, //maskBuffer
__global FPT *gdata1, //entryImageBuffer
__global FPT *gdata2, //gradient0
__global FPT *gdata3, //gradient1
__global FPT *gdata4, //gradient2
__global FPT *gdata5, //gradient3
__global FPT *gdata6, //hessian00
__global FPT *gdata7, //hessian01
__global FPT *gdata8, //hessian02
__global FPT *gdata9, //hessian03
__global FPT *gdata10, //hessian11
__global FPT *gdata11, //hessian12
__global FPT *gdata12, //hessian13
__global FPT *gdata13, //hessian22
__global FPT *gdata14, //hessian23
__global FPT *gdata15, //hessian33
__local volatile FPT* ldata, 
const int size)
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
    
    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata11[nIndex] + gdata11[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata11[nIndex];
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
        gdata11[nIndex] = ldata[nIndex];//transfer back to global memory
    }
    
    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata12[nIndex] + gdata12[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata12[nIndex];
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
        gdata12[nIndex] = ldata[nIndex];//transfer back to global memory
    }
    
    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata13[nIndex] + gdata13[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata13[nIndex];
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
        gdata13[nIndex] = ldata[nIndex];//transfer back to global memory
    }
    
    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata14[nIndex] + gdata14[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata14[nIndex];
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
        gdata14[nIndex] = ldata[nIndex];//transfer back to global memory
    }
    
    nrOfElems = size + (MODULO2(size));//now it's divisible by two
    divs = nrOfElems / 2;
    if(nIndex < divs)
    {
        if(nIndex + divs < size)
        {
            ldata[nIndex] = gdata15[nIndex] + gdata15[nIndex + divs];//linear addressing within a warp where divs is the stride
        }
        else
        {
            ldata[nIndex] = gdata15[nIndex];
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
        gdata15[nIndex] = ldata[nIndex];//transfer back to global memory
    }
}

__kernel void scaledRotationErrorWithGradAndHessBrent(const __global FPT *source,
const __global FPT *target,
const __global FPT *xGradient,
const __global FPT *yGradient,
__global FPT *grad0,
__global FPT *grad1,
__global FPT *grad2,
__global FPT *grad3,
__global FPT *hessian00,
__global FPT *hessian01,
__global FPT *hessian02,
__global FPT *hessian03,
__global FPT *hessian11,
__global FPT *hessian12,
__global FPT *hessian13,
__global FPT *hessian22,
__global FPT *hessian23,
__global FPT *hessian33,
__global FPT *diffout,
__global FPT *mask,
__local volatile FPT *lgrad0,
__local volatile FPT *lgrad1,
__local volatile FPT *lgrad2,
__local volatile FPT *lgrad3,
__local volatile FPT *lhessian00,
__local volatile FPT *lhessian01,
__local volatile FPT *lhessian02,
__local volatile FPT *lhessian03,
__local volatile FPT *lhessian11,
__local volatile FPT *lhessian12,
__local volatile FPT *lhessian13,
__local volatile FPT *lhessian22,
__local volatile FPT *lhessian23,
__local volatile FPT *lhessian33,
__local volatile FPT *ldiffout,
__local volatile FPT *lmask,
const int sourcewidth,
const int sourceheight,
const int targetwidth,
const int targetheight,
const FPT offsetx,
const FPT offsety,
const FPT cosangle,
const FPT negsinangle,
const FPT scale,
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
    lgrad3[nIndex] = Zero;
    lhessian00[nIndex] = Zero;
    lhessian01[nIndex] = Zero;
    lhessian02[nIndex] = Zero;
    lhessian03[nIndex] = Zero;
    lhessian11[nIndex] = Zero;
    lhessian12[nIndex] = Zero;
    lhessian13[nIndex] = Zero;
    lhessian22[nIndex] = Zero;
    lhessian23[nIndex] = Zero;
    lhessian33[nIndex] = Zero;
    ldiffout[nIndex] = Zero;
    lmask[nIndex] = Zero;

    //These vectors remain the same during the loops
    __private FPTTWO xvec = (FPTTWO)(cosangle,negsinangle) * scale;//warning: this is not the x vector but it is the vector added in the x direction
    __private FPTTWO yvec = (FPTTWO)(-xvec.y,xvec.x);//warning: this is not the y vector but it is the vector added in the y direction
    __private const int totalPixels = sourcewidth * sourceheight;
    while(i < totalPixels)
    {
        __private int column = i % sourcewidth;
        __private int row = (i - column)/sourcewidth;
        __private FPTTWO coord = (FPTTWO)(offsetx, offsety) + ((FPT)column) * xvec + ((FPT)row) * yvec;
        
        __private int4 xInterpolationIndxs;
        __private int4 yInterpolationIndxs;
        __private int4 combinedInterpolationIndices;
        __private FPT s;
        __private FPT diff;
        __private FPT Theta;
        //__private FPT j_scale;
        __private FPT j_logScale;
        
        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            lmask[nIndex] += One;
            xInterpolationIndxs = calculatexInterpolationIndxs(coord.x, doubleTargetWidth, targetwidth);
            yInterpolationIndxs = calculateyInterpolationIndxs(coord.y, doubleTargetHeight, targetheight, targetwidth);            
            s = interpolate(coord, xInterpolationIndxs, yInterpolationIndxs, target);
            diff = source[i] - s;
            //Theta = yGradient[i] * (FPT)column - xGradient[i] * (FPT)row;
            Theta = dot((FPTTWO)(yGradient[i], -xGradient[i]), (FPTTWO)((FPT)column, (FPT)row));
            
            /*
             * Switched to log scale for the scale parameter, which is more stable for optimization. 
             * The original implementation used a linear scale parameter, but this can lead to large 
             * updates and instability when the scale is far from 1. By optimizing in log space, 
             * we ensure that updates are multiplicative and more stable across a wide range of scales.
             */
            //j_scale = dot((FPTTWO)(xGradient[i], yGradient[i]), (FPTTWO)((FPT)column, (FPT)row));
            j_logScale = dot((FPTTWO)(xGradient[i], yGradient[i]), (FPTTWO)((FPT)column, (FPT)row)) * scale; //chain rule for log scale, this is the derivative of the error with respect to log(scale)

            //ldiffout[nIndex] += pown(diff,2);
            //lgrad0[nIndex] += diff * j_logScale;
            //lgrad1[nIndex] += diff * Theta;
            //lgrad2[nIndex] += diff * xGradient[nIndex];
            //lgrad3[nIndex] += diff * yGradient[nIndex];
            ldiffout[nIndex] += diff * diff;
            __private FPTFOUR tmp4 = fma((FPTFOUR)(j_logScale, Theta, xGradient[i], yGradient[i]), (FPTFOUR)diff, (FPTFOUR)(lgrad0[nIndex], lgrad1[nIndex], lgrad2[nIndex], lgrad3[nIndex])); //Slightly less accurate
            lgrad0[nIndex] = tmp4.x;
            lgrad1[nIndex] = tmp4.y;
            lgrad2[nIndex] = tmp4.z;
            lgrad3[nIndex] = tmp4.w;
            
            //lhessian00[nIndex] += pown(j_logScale,2); //this is more accurate
            //lhessian01[nIndex] += j_logScale * Theta;
            //lhessian02[nIndex] += j_logScale * xGradient[nIndex];
            //lhessian03[nIndex] += j_logScale * yGradient[nIndex];
            tmp4 = fma((FPTFOUR)(j_logScale, Theta, xGradient[i], yGradient[i]), (FPTFOUR)j_logScale, (FPTFOUR)(lhessian00[nIndex], lhessian01[nIndex], lhessian02[nIndex], lhessian03[nIndex])); //Slightly less accurate
            lhessian00[nIndex] = tmp4.x;
            lhessian01[nIndex] = tmp4.y;
            lhessian02[nIndex] = tmp4.z;
            lhessian03[nIndex] = tmp4.w;
            
            //lhessian11[nIndex] += pown(Theta,2); //this is more accurate
            //lhessian12[nIndex] += Theta * xGradient[nIndex];
            //lhessian13[nIndex] += Theta * yGradient[nIndex];
            __private FPTTHREE tmp = fma((FPTTHREE)(Theta, xGradient[i], yGradient[i]), (FPTTHREE)Theta, (FPTTHREE)(lhessian11[nIndex], lhessian12[nIndex], lhessian13[nIndex])); //Slightly less accurate
            lhessian11[nIndex] = tmp.x;
            lhessian12[nIndex] = tmp.y;
            lhessian13[nIndex] = tmp.z;
            
            //lhessian22[nIndex] += pown(xGradient[nIndex],2);
            //lhessian23[nIndex] += xGradient[nIndex] * yGradient[nIndex];
            //lhessian33[nIndex] += pown(yGradient[nIndex],2);
            tmp = fma(((FPTTHREE)(xGradient[i], xGradient[i], yGradient[i])), ((FPTTHREE)(xGradient[i], yGradient[i], yGradient[i])), (FPTTHREE)(lhessian22[nIndex], lhessian23[nIndex], lhessian33[nIndex])); //Slightly less accurate
            lhessian22[nIndex] = tmp.x;
            lhessian23[nIndex] = tmp.y;
            lhessian33[nIndex] = tmp.z;
        }
        // ensure reads are not out of bounds
        if(i + blockSize < totalPixels)
        {
            __private int lIdx = i + blockSize;
            column = lIdx % sourcewidth;
            row = (lIdx - column)/sourcewidth;
            coord = (FPTTWO)(offsetx, offsety) + ((FPT)column) * xvec + ((FPT)row) * yvec;            
            Msk = (int2)((int)round(coord.x), (int)round(coord.y));
            if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
            {
                lmask[nIndex] += One;
                xInterpolationIndxs = calculatexInterpolationIndxs(coord.x, doubleTargetWidth, targetwidth);
                yInterpolationIndxs = calculateyInterpolationIndxs(coord.y, doubleTargetHeight, targetheight, targetwidth); 
                s = interpolate(coord, xInterpolationIndxs, yInterpolationIndxs, target);
                diff = source[lIdx] - s;
                //Theta = yGradient[lIdx] * (FPT)column - xGradient[lIdx] * (FPT)row;
                Theta = dot((FPTTWO)(yGradient[lIdx], -xGradient[lIdx]), (FPTTWO)((FPT)column, (FPT)row));
                
                //j_scale = dot((FPTTWO)(xGradient[lIdx], yGradient[lIdx]), (FPTTWO)((FPT)column, (FPT)row));
                j_logScale = dot((FPTTWO)(xGradient[lIdx], yGradient[lIdx]), (FPTTWO)((FPT)column, (FPT)row)) * scale; //chain rule for log scale, this is the derivative of the error with respect to log(scale)
                
                //ldiffout[nIndex] += pown(diff,2);
                //lgrad0[nIndex] += diff * j_logScale;
                //lgrad1[nIndex] += diff * Theta;
                //lgrad2[nIndex] += diff * xGradient[lIdx];
                //lgrad3[nIndex] += diff * yGradient[lIdx];
                ldiffout[nIndex] += diff * diff;
                __private FPTFOUR tmp4 = fma((FPTFOUR)(j_logScale, Theta, xGradient[lIdx], yGradient[lIdx]), (FPTFOUR)diff, (FPTFOUR)(lgrad0[nIndex], lgrad1[nIndex], lgrad2[nIndex], lgrad3[nIndex])); //Slightly less accurate
                lgrad0[nIndex] = tmp4.x;
                lgrad1[nIndex] = tmp4.y;
                lgrad2[nIndex] = tmp4.z;
                lgrad3[nIndex] = tmp4.w;
                
                //lhessian00[nIndex] += pown(j_logScale,2); //this is more accurate
                //lhessian01[nIndex] += j_logScale * Theta;
                //lhessian02[nIndex] += j_logScale * xGradient[lIdx];
                //lhessian03[nIndex] += j_logScale * yGradient[lIdx];
                tmp4 = fma((FPTFOUR)(j_logScale, Theta, xGradient[lIdx], yGradient[lIdx]), (FPTFOUR)j_logScale, (FPTFOUR)(lhessian00[nIndex], lhessian01[nIndex], lhessian02[nIndex], lhessian03[nIndex])); //Slightly less accurate
                lhessian00[nIndex] = tmp4.x;
                lhessian01[nIndex] = tmp4.y;
                lhessian02[nIndex] = tmp4.z;
                lhessian03[nIndex] = tmp4.w;
                
                //lhessian11[nIndex] += pown(Theta,2); //this is more accurate
                //lhessian12[nIndex] += Theta * xGradient[lIdx];
                //lhessian13[nIndex] += Theta * yGradient[lIdx];
                __private FPTTHREE tmp = fma((FPTTHREE)(Theta, xGradient[lIdx], yGradient[lIdx]), (FPTTHREE)Theta, (FPTTHREE)(lhessian11[nIndex], lhessian12[nIndex], lhessian13[nIndex])); //Slightly less accurate
                lhessian11[nIndex] = tmp.x;
                lhessian12[nIndex] = tmp.y;
                lhessian13[nIndex] = tmp.z;
                
                //lhessian22[nIndex] += pown(xGradient[lIdx],2);
                //lhessian23[nIndex] += xGradient[lIdx] * yGradient[lIdx];
                //lhessian33[nIndex] += pown(yGradient[lIdx],2);
                tmp = fma(((FPTTHREE)(xGradient[lIdx], xGradient[lIdx], yGradient[lIdx])), ((FPTTHREE)(xGradient[lIdx], yGradient[lIdx], yGradient[lIdx])), (FPTTHREE)(lhessian22[nIndex], lhessian23[nIndex], lhessian33[nIndex])); //Slightly less accurate
                lhessian22[nIndex] = tmp.x;
                lhessian23[nIndex] = tmp.y;
                lhessian33[nIndex] = tmp.z;
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
            lgrad3[nIndex] += lgrad3[nIndex + i];
            lhessian00[nIndex] += lhessian00[nIndex + i];
            lhessian01[nIndex] += lhessian01[nIndex + i];
            lhessian02[nIndex] += lhessian02[nIndex + i];
            lhessian03[nIndex] += lhessian03[nIndex + i];
            lhessian11[nIndex] += lhessian11[nIndex + i];
            lhessian12[nIndex] += lhessian12[nIndex + i];
            lhessian13[nIndex] += lhessian13[nIndex + i];
            lhessian22[nIndex] += lhessian22[nIndex + i];
            lhessian23[nIndex] += lhessian23[nIndex + i];
            lhessian33[nIndex] += lhessian33[nIndex + i];
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
        grad3[i] = lgrad3[0];
        hessian00[i] = lhessian00[0];
        hessian01[i] = lhessian01[0];
        hessian02[i] = lhessian02[0];
        hessian03[i] = lhessian03[0];
        hessian11[i] = lhessian11[0];
        hessian12[i] = lhessian12[0];
        hessian13[i] = lhessian13[0];
        hessian22[i] = lhessian22[0];
        hessian23[i] = lhessian23[0];
        hessian33[i] = lhessian33[0];
        diffout[i] = ldiffout[0];
        mask[i] = lmask[0];
    }
}

__kernel void scaledRotationTransformImageWithBsplineInterpolation(const __global FPT *source ,__global FPT *target, const int sourcewidth, const int sourceheight, const int doubleSourceWidth, const int doubleSourceHeight, const FPT offsetx, const FPT offsety, const FPT cosangle, const FPT negsinangle, const FPT scale)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !SOURCE! pixel
    if(nIndex < sourcewidth * sourceheight)
    {
        __private int column = nIndex % sourcewidth;
        __private int row = (nIndex - column)/sourcewidth;

        __private FPTTWO xvec = (FPTTWO)(cosangle, negsinangle) * scale;//warning: this is not the x vector but it is the vector added in the x direction
        __private FPTTWO yvec = (FPTTWO)(-xvec.y, xvec.x);//warning: this is not the y vector but it is the vector added in the y direction
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

__kernel void resizingScaledRotationTransformImageWithBsplineInterpolation(const __global FPT *source ,
__global FPT *target, 
const int sourcewidth, 
const int sourceheight, 
const int doubleSourceWidth, 
const int doubleSourceHeight, 
const int targetwidth, 
const int targetheight, 
const FPT offsetx, 
const FPT offsety, 
const FPT cosangle, 
const FPT negsinangle,
const FPT scale)
{
    __private int nIndex = get_global_id(0);//this directly corresponds to the linear address of the !TARGET! pixel
    if(nIndex < targetwidth * targetheight)
    {
        __private int column = nIndex % targetwidth;
        __private int row = (nIndex - column)/targetwidth;

        __private FPTTWO xvec = (FPTTWO)(cosangle, negsinangle) * scale;//warning: this is not the x vector but it is the vector added in the x direction
        __private FPTTWO yvec = (FPTTWO)(-xvec.y, xvec.x);//warning: this is not the y vector but it is the vector added in the y direction
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
#endif


#ifdef AFFINE
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


__kernel void sumInLocalMemoryAffineCombined(__global FPT *gdata0, //maskBuffer
__global FPT *gdata1, //entryImageBuffer
__global FPT *gdata2, //gradient0
__global FPT *gdata3, //gradient1
__global FPT *gdata4, //gradient2
__global FPT *gdata5, //gradient3
__global FPT *gdata6, //gradient4
__global FPT *gdata7, //gradient5
__global FPT *gdata8, //hessian00
__global FPT *gdata9, //hessian01
__global FPT *gdata10, //hessian02
__global FPT *gdata11, //hessian03
__global FPT *gdata12, //hessian04
__global FPT *gdata13, //hessian05
__global FPT *gdata14, //hessian11
__global FPT *gdata15, //hessian12
__global FPT *gdata16, //hessian13
__global FPT *gdata17, //hessian14
__global FPT *gdata18, //hessian15
__global FPT *gdata19, //hessian22
__global FPT *gdata20, //hessian23
__global FPT *gdata21, //hessian24
__global FPT *gdata22, //hessian25
__global FPT *gdata23, //hessian33
__global FPT *gdata24, //hessian34
__global FPT *gdata25, //hessian35
__global FPT *gdata26, //hessian44
__global FPT *gdata27, //hessian45
__global FPT *gdata28, //hessian55
__local volatile FPT* ldata, 
const int size)
{
	__global FPT* gdata[29] = {
	gdata0, 
	gdata1, 
    gdata2,
    gdata3,
    gdata4,
    gdata5,
    gdata6,
    gdata7,
    gdata8,
    gdata9,
    gdata10,
    gdata11,
    gdata12,
    gdata13,
    gdata14,
    gdata15,
    gdata16,
    gdata17,
    gdata18,
    gdata19,
    gdata20,
    gdata21,
    gdata22,
    gdata23,
    gdata24,
    gdata25,
    gdata26,
    gdata27,
    gdata28};
    /*
    Only a single workgroup must be started but this won't be tested here.
    The number of threads MUST be at least (size + (size % 2))/2!!!!!
    Watch out for the barriers. A barrier must be reached by all threads,
    therefore you can't use a global return statement to get rid of them,
    rather you just define behavior for the threads you want to be active
    and then let all threads hit the barriers.
    */
    
    __private int nIndex = get_local_id(0);
    __private int nrOfElems;
    __private int divs;
    for(int i = 0; i < 29; i++)
    {
	    nrOfElems = size + (MODULO2(size));//now it's divisible by two
	    divs = nrOfElems / 2;
	    if(nIndex < divs)
	    {
	        if(nIndex + divs < size)
	        {
	            ldata[nIndex] = gdata[i][nIndex] + gdata[i][nIndex + divs];//linear addressing within a warp where divs is the stride
	        }
	        else
	        {
	            ldata[nIndex] = gdata[i][nIndex];
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
	        gdata[i][nIndex] = ldata[nIndex];//transfer back to global memory
	    }
    }
}

__kernel void affineErrorWithGradAndHess(const __global FPT *source,
const __global FPT *target,
const __global FPT *xGradient,
const __global FPT *yGradient,
__global FPT *mask,
__global FPT *grad0,
__global FPT *grad1,
__global FPT *grad2,
__global FPT *grad3,
__global FPT *grad4,
__global FPT *grad5,
__global FPT *hessian00,
__global FPT *hessian01,
__global FPT *hessian02,
__global FPT *hessian03,
__global FPT *hessian04,
__global FPT *hessian05,
__global FPT *hessian11,
__global FPT *hessian12,
__global FPT *hessian13,
__global FPT *hessian14,
__global FPT *hessian15,
__global FPT *hessian22,
__global FPT *hessian23,
__global FPT *hessian24,
__global FPT *hessian25,
__global FPT *hessian33,
__global FPT *hessian34,
__global FPT *hessian35,
__global FPT *hessian44,
__global FPT *hessian45,
__global FPT *hessian55,
__global FPT *diffout,
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
            __private FPT diff = source[nIndex] - s;
            __private FPTTWO dx = (FPTTWO)((FPT)column, (FPT)row) * xGradient[nIndex];
            __private FPTTWO dy = (FPTTWO)((FPT)column, (FPT)row) * yGradient[nIndex];
            __private FPTTWO dxy = (FPTTWO)(xGradient[nIndex], yGradient[nIndex]);

            __private FPTEIGHT tmp8 = (FPTEIGHT)(diff, dx.x, dx.y, dy.x, dy.y, dxy.x, dxy.y, Zero) * diff; //Slightly less accurate
            diffout[nIndex] = tmp8.s0;
            
            grad0[nIndex] = tmp8.s1;
            grad1[nIndex] = tmp8.s2;
            grad2[nIndex] = tmp8.s3;
            grad3[nIndex] = tmp8.s4;
            grad4[nIndex] = tmp8.s5;
            grad5[nIndex] = tmp8.s6;
            //hessian00[nIndex] = pown(dx0,2); //this is more accurate
            //hessian01[nIndex] = dx0 * dx1;
            //hessian02[nIndex] = dx0 * dy0;
            //hessian03[nIndex] = dx0 * dy1;
            //hessian04[nIndex] = dx0 * dx;
            //hessian05[nIndex] = dx0 * dy;
            __private FPTTHREE tmp3 = (FPTTHREE)(dx.x, dx.y, dy.x) * dx.x; //Slightly less accurate
            hessian00[nIndex] = tmp3.x;
            hessian01[nIndex] = tmp3.y;
            hessian02[nIndex] = tmp3.z;
            tmp3 = (FPTTHREE)(dy.y, dxy.x, dxy.y) * dx.x; //Slightly less accurate
            hessian03[nIndex] = tmp3.x;
            hessian04[nIndex] = tmp3.y;
            hessian05[nIndex] = tmp3.z;
            
            //hessian11[nIndex] = pown(dx1,2); //this is more accurate
            //hessian12[nIndex] = dx1 * dy0;
            //hessian13[nIndex] = dx1 * dy1;
            //hessian14[nIndex] = dx1 * dx;
            //hessian15[nIndex] = dx1 * dy;
            hessian11[nIndex] = dx.y * dx.y; //Slightly less accurate
            FPTFOUR tmp4 = (FPTFOUR)(dy.x, dy.y, dxy.x, dxy.y) * dx.y;
            hessian12[nIndex] = tmp4.x;
            hessian13[nIndex] = tmp4.y;
            hessian14[nIndex] = tmp4.z;
            hessian15[nIndex] = tmp4.w;
            
            //hessian22[nIndex] = pown(dy0,2);
            //hessian23[nIndex] = dy0 * dy1;
            //hessian24[nIndex] = dy0 * dx;
            //hessian25[nIndex] = dy0 * dy;
            tmp4 = (FPTFOUR)(dy.x, dy.y, dxy.x, dxy.y) * dy.x; //Slightly less accurate
            hessian22[nIndex] = tmp4.x;
            hessian23[nIndex] = tmp4.y;
            hessian24[nIndex] = tmp4.z;
            hessian25[nIndex] = tmp4.w;
            
            //hessian33[nIndex] = pown(dy1,2);
            //hessian34[nIndex] = dy1 * dx;
            //hessian35[nIndex] = dy1 * dy;
            tmp3 = (FPTTHREE)(dy.y, dxy.x, dxy.y) * dy.y; //Slightly less accurate
            hessian33[nIndex] = tmp3.x;
            hessian34[nIndex] = tmp3.y;
            hessian35[nIndex] = tmp3.z;
            
            //hessian44[nIndex] = pown(dx,2);
            //hessian45[nIndex] = dx * dy;
            //hessian55[nIndex] = pown(dy,2);
            tmp3 = (FPTTHREE)(dxy.x, dxy.y, dxy.y) * (FPTTHREE)(dxy.x, dxy.x, dxy.y); //Slightly less accurate
            hessian44[nIndex] = tmp3.x;
            hessian45[nIndex] = tmp3.y;
            hessian55[nIndex] = tmp3.z;
        }
    }
}

__kernel void affineErrorWithGradAndHessBrent(const __global FPT *source,
const __global FPT *target,
const __global FPT *xGradient,
const __global FPT *yGradient,
__global FPT *grad0,
__global FPT *grad1,
__global FPT *grad2,
__global FPT *grad3,
__global FPT *grad4,
__global FPT *grad5,
__global FPT *hessian00,
__global FPT *hessian01,
__global FPT *hessian02,
__global FPT *hessian03,
__global FPT *hessian04,
__global FPT *hessian05,
__global FPT *hessian11,
__global FPT *hessian12,
__global FPT *hessian13,
__global FPT *hessian14,
__global FPT *hessian15,
__global FPT *hessian22,
__global FPT *hessian23,
__global FPT *hessian24,
__global FPT *hessian25,
__global FPT *hessian33,
__global FPT *hessian34,
__global FPT *hessian35,
__global FPT *hessian44,
__global FPT *hessian45,
__global FPT *hessian55,
__global FPT *diffout,
__global FPT *mask,
__local volatile FPT *lgrad0,
__local volatile FPT *lgrad1,
__local volatile FPT *lgrad2,
__local volatile FPT *lgrad3,
__local volatile FPT *lgrad4,
__local volatile FPT *lgrad5,
__local volatile FPT *lhessian00,
__local volatile FPT *lhessian01,
__local volatile FPT *lhessian02,
__local volatile FPT *lhessian03,
__local volatile FPT *lhessian04,
__local volatile FPT *lhessian05,
__local volatile FPT *lhessian11,
__local volatile FPT *lhessian12,
__local volatile FPT *lhessian13,
__local volatile FPT *lhessian14,
__local volatile FPT *lhessian15,
__local volatile FPT *lhessian22,
__local volatile FPT *lhessian23,
__local volatile FPT *lhessian24,
__local volatile FPT *lhessian25,
__local volatile FPT *lhessian33,
__local volatile FPT *lhessian34,
__local volatile FPT *lhessian35,
__local volatile FPT *lhessian44,
__local volatile FPT *lhessian45,
__local volatile FPT *lhessian55,
__local volatile FPT *ldiffout,
__local volatile FPT *lmask,
const int sourcewidth,
const int sourceheight,
const int targetwidth,
const int targetheight,
const FPT offsetx, 
const FPT offsety, 
const FPT a11, 
const FPT a12, 
const FPT a21, 
const FPT a22,
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
    lgrad3[nIndex] = Zero;
    lgrad4[nIndex] = Zero;
    lgrad5[nIndex] = Zero;
    lhessian00[nIndex] = Zero;
    lhessian01[nIndex] = Zero;
    lhessian02[nIndex] = Zero;
    lhessian03[nIndex] = Zero;
    lhessian04[nIndex] = Zero;
    lhessian05[nIndex] = Zero;
    lhessian11[nIndex] = Zero;
    lhessian12[nIndex] = Zero;
    lhessian13[nIndex] = Zero;
    lhessian14[nIndex] = Zero;
    lhessian15[nIndex] = Zero;
    lhessian22[nIndex] = Zero;
    lhessian23[nIndex] = Zero;
    lhessian24[nIndex] = Zero;
    lhessian25[nIndex] = Zero;
    lhessian33[nIndex] = Zero;
    lhessian34[nIndex] = Zero;
    lhessian35[nIndex] = Zero;
    lhessian44[nIndex] = Zero;
    lhessian45[nIndex] = Zero;
    lhessian55[nIndex] = Zero;
    ldiffout[nIndex] = Zero;
    lmask[nIndex] = Zero;

    //These vectors remain the same during the loops
    __private FPTTWO xvec = (FPTTWO)(a11,a21);//warning: this is not the x vector but it is the vector added in the x direction
    __private FPTTWO yvec = (FPTTWO)(a12,a22);//warning: this is not the y vector but it is the vector added in the y direction
    __private const int totalPixels = sourcewidth * sourceheight;
    while(i < totalPixels)
    {
        __private int column = i % sourcewidth;
        __private int row = (i - column)/sourcewidth;
        __private FPTTWO coord = (FPTTWO)(offsetx, offsety) + ((FPT)column) * xvec + ((FPT)row) * yvec;
        
        __private int4 xInterpolationIndxs;
        __private int4 yInterpolationIndxs;
        __private int4 combinedInterpolationIndices;
        __private FPT s;
        __private FPT diff;
        __private FPTTWO dx;
        __private FPTTWO dy;
        __private FPTTWO dxy;
        
        __private int2 Msk = (int2)((int)round(coord.x), (int)round(coord.y));
        if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
        {
            lmask[nIndex] += One;
            xInterpolationIndxs = calculatexInterpolationIndxs(coord.x, doubleTargetWidth, targetwidth);
            yInterpolationIndxs = calculateyInterpolationIndxs(coord.y, doubleTargetHeight, targetheight, targetwidth);            
            s = interpolate(coord, xInterpolationIndxs, yInterpolationIndxs, target);
            diff = source[i] - s;

            dx = (FPTTWO)((FPT)column, (FPT)row) * xGradient[i];
            dy = (FPTTWO)((FPT)column, (FPT)row) * yGradient[i];
            dxy = (FPTTWO)(xGradient[i], yGradient[i]);
            
            //ldiffout[nIndex] += pown(diff,2);
            //lgrad0[nIndex] += diff * dx0;
            //lgrad1[nIndex] += diff * dx1;
            //lgrad2[nIndex] += diff * dy0;
            //lgrad3[nIndex] += diff * dy1;
            //lgrad4[nIndex] += diff * dx;
            //lgrad5[nIndex] += diff * dy;
            __private FPTEIGHT tmp8 = fma((FPTEIGHT)(diff, dx.x, dx.y, dy.x, dy.y, dxy.x, dxy.y, Zero), (FPTEIGHT)diff, (FPTEIGHT)(ldiffout[nIndex], lgrad0[nIndex], lgrad1[nIndex], lgrad2[nIndex], lgrad3[nIndex], lgrad4[nIndex], lgrad5[nIndex], Zero)); //Slightly less accurate
            ldiffout[nIndex] = tmp8.s0;
            lgrad0[nIndex] = tmp8.s1;
            lgrad1[nIndex] = tmp8.s2;
            lgrad2[nIndex] = tmp8.s3;
            lgrad3[nIndex] = tmp8.s4;
            lgrad4[nIndex] = tmp8.s5;
            lgrad5[nIndex] = tmp8.s6;
            //lhessian00[nIndex] += pown(dx0,2); //this is more accurate
            //lhessian01[nIndex] += dx0 * dx1;
            //lhessian02[nIndex] += dx0 * dy0;
            //lhessian03[nIndex] += dx0 * dy1;
            //lhessian04[nIndex] += dx0 * dx;
            //lhessian05[nIndex] += dx0 * dy;
            __private FPTTHREE tmp3 = fma((FPTTHREE)(dx.x, dx.y, dy.x), (FPTTHREE)dx.x, (FPTTHREE)(lhessian00[nIndex], lhessian01[nIndex], lhessian02[nIndex])); //Slightly less accurate
            lhessian00[nIndex] = tmp3.x;
            lhessian01[nIndex] = tmp3.y;
            lhessian02[nIndex] = tmp3.z;
            tmp3 = fma((FPTTHREE)(dy.y, dxy.x, dxy.y), (FPTTHREE)dx.x, (FPTTHREE)(lhessian03[nIndex], lhessian04[nIndex], lhessian05[nIndex])); //Slightly less accurate
            lhessian03[nIndex] = tmp3.x;
            lhessian04[nIndex] = tmp3.y;
            lhessian05[nIndex] = tmp3.z;
            
            //lhessian11[nIndex] += pown(dx1,2); //this is more accurate
            //lhessian12[nIndex] += dx1 * dy0;
            //lhessian13[nIndex] += dx1 * dy1;
            //lhessian14[nIndex] += dx1 * dx;
            //lhessian15[nIndex] += dx1 * dy;
            lhessian11[nIndex] += dx.y * dx.y; //Slightly less accurate
            FPTFOUR tmp4 = fma((FPTFOUR)(dy.x, dy.y, dxy.x, dxy.y), (FPTFOUR)dx.y, (FPTFOUR)(lhessian12[nIndex], lhessian13[nIndex], lhessian14[nIndex], lhessian15[nIndex])); //Slightly less accurate
            lhessian12[nIndex] = tmp4.x;
            lhessian13[nIndex] = tmp4.y;
            lhessian14[nIndex] = tmp4.z;
            lhessian15[nIndex] = tmp4.w;
            
            //lhessian22[nIndex] += pown(dy0,2);
            //lhessian23[nIndex] += dy0 * dy1;
            //lhessian24[nIndex] += dy0 * dx;
            //lhessian25[nIndex] += dy0 * dy;
            tmp4 = fma((FPTFOUR)(dy.x, dy.y, dxy.x, dxy.y), (FPTFOUR)dy.x, (FPTFOUR)(lhessian22[nIndex], lhessian23[nIndex], lhessian24[nIndex], lhessian25[nIndex])); //Slightly less accurate
            lhessian22[nIndex] = tmp4.x;
            lhessian23[nIndex] = tmp4.y;
            lhessian24[nIndex] = tmp4.z;
            lhessian25[nIndex] = tmp4.w;
            
            //lhessian33[nIndex] += pown(dy1,2);
            //lhessian34[nIndex] += dy1 * dx;
            //lhessian35[nIndex] += dy1 * dy;
            tmp3 = fma((FPTTHREE)(dy.y, dxy.x, dxy.y), (FPTTHREE)dy.y, (FPTTHREE)(lhessian33[nIndex], lhessian34[nIndex], lhessian35[nIndex])); //Slightly less accurate
            lhessian33[nIndex] = tmp3.x;
            lhessian34[nIndex] = tmp3.y;
            lhessian35[nIndex] = tmp3.z;
            
            //lhessian44[nIndex] += pown(dx,2);
            //lhessian45[nIndex] += dx * dy;
            //lhessian55[nIndex] += pown(dy,2);
            tmp3 = fma((FPTTHREE)(dxy.x, dxy.y, dxy.y), (FPTTHREE)(dxy.x, dxy.x, dxy.y), (FPTTHREE)(lhessian44[nIndex], lhessian45[nIndex], lhessian55[nIndex])); //Slightly less accurate
            lhessian44[nIndex] = tmp3.x;
            lhessian45[nIndex] = tmp3.y;
            lhessian55[nIndex] = tmp3.z;
        }
        // ensure reads are not out of bounds
        if(i + blockSize < totalPixels)
        {
            __private int lIdx = i + blockSize;
            column = lIdx % sourcewidth;
            row = (lIdx - column)/sourcewidth;
            coord = (FPTTWO)(offsetx, offsety) + ((FPT)column) * xvec + ((FPT)row) * yvec;            
            Msk = (int2)((int)round(coord.x), (int)round(coord.y));
            if ((Msk.x >= 0) && (Msk.x < targetwidth) && (Msk.y >= 0) && (Msk.y < targetheight))
            {
                lmask[nIndex] += One;
                xInterpolationIndxs = calculatexInterpolationIndxs(coord.x, doubleTargetWidth, targetwidth);
                yInterpolationIndxs = calculateyInterpolationIndxs(coord.y, doubleTargetHeight, targetheight, targetwidth); 
                s = interpolate(coord, xInterpolationIndxs, yInterpolationIndxs, target);
                diff = source[lIdx] - s;

                dx = (FPTTWO)((FPT)column, (FPT)row) * xGradient[lIdx];
                dy = (FPTTWO)((FPT)column, (FPT)row) * yGradient[lIdx];
                dxy = (FPTTWO)(xGradient[lIdx], yGradient[lIdx]);
                
                //ldiffout[nIndex] += pown(diff,2);
                //lgrad0[nIndex] += diff * dx0;
                //lgrad1[nIndex] += diff * dx1;
                //lgrad2[nIndex] += diff * dy0;
                //lgrad3[nIndex] += diff * dy1;
                //lgrad4[nIndex] += diff * dx;
                //lgrad5[nIndex] += diff * dy;
                __private FPTEIGHT tmp8 = fma((FPTEIGHT)(diff, dx.x, dx.y, dy.x, dy.y, dxy.x, dxy.y, Zero), (FPTEIGHT)diff, (FPTEIGHT)(ldiffout[nIndex], lgrad0[nIndex], lgrad1[nIndex], lgrad2[nIndex], lgrad3[nIndex], lgrad4[nIndex], lgrad5[nIndex], Zero)); //Slightly less accurate
                ldiffout[nIndex] = tmp8.s0;
                lgrad0[nIndex] = tmp8.s1;
                lgrad1[nIndex] = tmp8.s2;
                lgrad2[nIndex] = tmp8.s3;
                lgrad3[nIndex] = tmp8.s4;
                lgrad4[nIndex] = tmp8.s5;
                lgrad5[nIndex] = tmp8.s6;
                //lhessian00[nIndex] += pown(dx0,2); //this is more accurate
                //lhessian01[nIndex] += dx0 * dx1;
                //lhessian02[nIndex] += dx0 * dy0;
                //lhessian03[nIndex] += dx0 * dy1;
                //lhessian04[nIndex] += dx0 * dx;
                //lhessian05[nIndex] += dx0 * dy;
                __private FPTTHREE tmp3 = fma((FPTTHREE)(dx.x, dx.y, dy.x), (FPTTHREE)dx.x, (FPTTHREE)(lhessian00[nIndex], lhessian01[nIndex], lhessian02[nIndex])); //Slightly less accurate
                lhessian00[nIndex] = tmp3.x;
                lhessian01[nIndex] = tmp3.y;
                lhessian02[nIndex] = tmp3.z;
                tmp3 = fma((FPTTHREE)(dy.y, dxy.x, dxy.y), (FPTTHREE)dx.x, (FPTTHREE)(lhessian03[nIndex], lhessian04[nIndex], lhessian05[nIndex])); //Slightly less accurate
                lhessian03[nIndex] = tmp3.x;
                lhessian04[nIndex] = tmp3.y;
                lhessian05[nIndex] = tmp3.z;
                
                //lhessian11[nIndex] += pown(dx1,2); //this is more accurate
                //lhessian12[nIndex] += dx1 * dy0;
                //lhessian13[nIndex] += dx1 * dy1;
                //lhessian14[nIndex] += dx1 * dx;
                //lhessian15[nIndex] += dx1 * dy;
                lhessian11[nIndex] += dx.y * dx.y; //Slightly less accurate
                FPTFOUR tmp4 = fma((FPTFOUR)(dy.x, dy.y, dxy.x, dxy.y), (FPTFOUR)dx.y, (FPTFOUR)(lhessian12[nIndex], lhessian13[nIndex], lhessian14[nIndex], lhessian15[nIndex])); //Slightly less accurate
                lhessian12[nIndex] = tmp4.x;
                lhessian13[nIndex] = tmp4.y;
                lhessian14[nIndex] = tmp4.z;
                lhessian15[nIndex] = tmp4.w;
                
                //lhessian22[nIndex] += pown(dy0,2);
                //lhessian23[nIndex] += dy0 * dy1;
                //lhessian24[nIndex] += dy0 * dx;
                //lhessian25[nIndex] += dy0 * dy;
                tmp4 = fma((FPTFOUR)(dy.x, dy.y, dxy.x, dxy.y), (FPTFOUR)dy.x, (FPTFOUR)(lhessian22[nIndex], lhessian23[nIndex], lhessian24[nIndex], lhessian25[nIndex])); //Slightly less accurate
                lhessian22[nIndex] = tmp4.x;
                lhessian23[nIndex] = tmp4.y;
                lhessian24[nIndex] = tmp4.z;
                lhessian25[nIndex] = tmp4.w;
                
                //lhessian33[nIndex] += pown(dy1,2);
                //lhessian34[nIndex] += dy1 * dx;
                //lhessian35[nIndex] += dy1 * dy;
                tmp3 = fma((FPTTHREE)(dy.y, dxy.x, dxy.y), (FPTTHREE)dy.y, (FPTTHREE)(lhessian33[nIndex], lhessian34[nIndex], lhessian35[nIndex])); //Slightly less accurate
                lhessian33[nIndex] = tmp3.x;
                lhessian34[nIndex] = tmp3.y;
                lhessian35[nIndex] = tmp3.z;
                
                //lhessian44[nIndex] += pown(dx,2);
                //lhessian45[nIndex] += dx * dy;
                //lhessian55[nIndex] += pown(dy,2);
                tmp3 = fma((FPTTHREE)(dxy.x, dxy.y, dxy.y), (FPTTHREE)(dxy.x, dxy.x, dxy.y), (FPTTHREE)(lhessian44[nIndex], lhessian45[nIndex], lhessian55[nIndex])); //Slightly less accurate
                lhessian44[nIndex] = tmp3.x;
                lhessian45[nIndex] = tmp3.y;
                lhessian55[nIndex] = tmp3.z;
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
            lgrad3[nIndex] += lgrad3[nIndex + i];
            lgrad4[nIndex] += lgrad4[nIndex + i];
            lgrad5[nIndex] += lgrad5[nIndex + i];
            lhessian00[nIndex] += lhessian00[nIndex + i];
            lhessian01[nIndex] += lhessian01[nIndex + i];
            lhessian02[nIndex] += lhessian02[nIndex + i];
            lhessian03[nIndex] += lhessian03[nIndex + i];
            lhessian04[nIndex] += lhessian04[nIndex + i];
            lhessian05[nIndex] += lhessian05[nIndex + i];
            lhessian11[nIndex] += lhessian11[nIndex + i];
            lhessian12[nIndex] += lhessian12[nIndex + i];
            lhessian13[nIndex] += lhessian13[nIndex + i];
            lhessian14[nIndex] += lhessian14[nIndex + i];
            lhessian15[nIndex] += lhessian15[nIndex + i];
            lhessian22[nIndex] += lhessian22[nIndex + i];
            lhessian23[nIndex] += lhessian23[nIndex + i];
            lhessian24[nIndex] += lhessian24[nIndex + i];
            lhessian25[nIndex] += lhessian25[nIndex + i];
            lhessian33[nIndex] += lhessian33[nIndex + i];
            lhessian34[nIndex] += lhessian34[nIndex + i];
            lhessian35[nIndex] += lhessian35[nIndex + i];
            lhessian44[nIndex] += lhessian44[nIndex + i];
            lhessian45[nIndex] += lhessian45[nIndex + i];
            lhessian55[nIndex] += lhessian55[nIndex + i];
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
        grad3[i] = lgrad3[0];
        grad4[i] = lgrad4[0];
        grad5[i] = lgrad5[0];
        hessian00[i] = lhessian00[0];
        hessian01[i] = lhessian01[0];
        hessian02[i] = lhessian02[0];
        hessian03[i] = lhessian03[0];
        hessian04[i] = lhessian04[0];
        hessian05[i] = lhessian05[0];
        hessian11[i] = lhessian11[0];
        hessian12[i] = lhessian12[0];
        hessian13[i] = lhessian13[0];
        hessian14[i] = lhessian14[0];
        hessian15[i] = lhessian15[0];
        hessian22[i] = lhessian22[0];
        hessian23[i] = lhessian23[0];
        hessian24[i] = lhessian24[0];
        hessian25[i] = lhessian25[0];
        hessian33[i] = lhessian33[0];
        hessian34[i] = lhessian34[0];
        hessian35[i] = lhessian35[0];
        hessian44[i] = lhessian44[0];
        hessian45[i] = lhessian45[0];
        hessian55[i] = lhessian55[0];
        diffout[i] = ldiffout[0];
        mask[i] = lmask[0];
    }
}
#endif

#if defined(HYBRID) && defined(USE_DOUBLE)
__kernel void ConvertDoubleToFloat(__global const double *source, __global float *target, const int size)
{
    __private int nIndex = get_global_id(0); // this directly corresponds to the offset
    if(nIndex < size)
    {
        target[nIndex] = (float)(source[nIndex]); // There is a bug in my version of the NVidia driver preventing me from using convert_float_sat_rte
    }
}
#endif
