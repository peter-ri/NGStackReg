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



__kernel void sumInLocalMemoryAffineCombined(__global FPT *gdata0,
__global FPT *gdata1,
__global FPT *gdata2,
__global FPT *gdata3,
__global FPT *gdata4,
__global FPT *gdata5,
__global FPT *gdata6,
__global FPT *gdata7,
__global FPT *gdata8,
__global FPT *gdata9,
__global FPT *gdata10,
__global FPT *gdata11,
__global FPT *gdata12,
__global FPT *gdata13,
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
}

__kernel void affineErrorWithGradAndHess(const __global FPT *source,
const __global FPT *target,
const __global FPT *xGradient,
const __global FPT *yGradient,
__global FPT *mask,
__global FPT *grad1,
__global FPT *grad3,
__global FPT *grad5,
__global FPT *hessian01,
__global FPT *hessian03,
__global FPT *hessian05,
__global FPT *hessian12,
__global FPT *hessian14,
__global FPT *hessian22,
__global FPT *hessian24,
__global FPT *hessian33,
__global FPT *hessian35,
__global FPT *hessian45,
__global FPT *diffout,
const int secondBufferOffset,
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
            
            mask[secondBufferOffset + nIndex] = tmp8.s1;
            grad1[nIndex] = tmp8.s2;
            grad1[secondBufferOffset + nIndex] = tmp8.s3;
            grad3[nIndex] = tmp8.s4;
            grad3[secondBufferOffset + nIndex] = tmp8.s5;
            grad5[nIndex] = tmp8.s6;
            //hessian00[nIndex] = pown(dx0,2); //this is more accurate
            //hessian01[nIndex] = dx0 * dx1;
            //hessian02[nIndex] = dx0 * dy0;
            //hessian03[nIndex] = dx0 * dy1;
            //hessian04[nIndex] = dx0 * dx;
            //hessian05[nIndex] = dx0 * dy;
            __private FPTTHREE tmp3 = (FPTTHREE)(dx.x, dx.y, dy.x) * dx.x; //Slightly less accurate
            grad5[secondBufferOffset + nIndex] = tmp3.x;
            hessian01[nIndex] = tmp3.y;
            hessian01[secondBufferOffset + nIndex] = tmp3.z;
            tmp3 = (FPTTHREE)(dy.y, dxy.x, dxy.y) * dx.x; //Slightly less accurate
            hessian03[nIndex] = tmp3.x;
            hessian03[secondBufferOffset + nIndex] = tmp3.y;
            hessian05[nIndex] = tmp3.z;
            
            //hessian11[nIndex] = pown(dx1,2); //this is more accurate
            //hessian12[nIndex] = dx1 * dy0;
            //hessian13[nIndex] = dx1 * dy1;
            //hessian14[nIndex] = dx1 * dx;
            //hessian15[nIndex] = dx1 * dy;
            hessian05[secondBufferOffset + nIndex] = dx.y * dx.y; //Slightly less accurate
            FPTFOUR tmp4 = (FPTFOUR)(dy.x, dy.y, dxy.x, dxy.y) * dx.y;
            hessian12[nIndex] = tmp4.x;
            hessian12[secondBufferOffset + nIndex] = tmp4.y;
            hessian14[nIndex] = tmp4.z;
            hessian14[secondBufferOffset + nIndex] = tmp4.w;
            
            //hessian22[nIndex] = pown(dy0,2);
            //hessian23[nIndex] = dy0 * dy1;
            //hessian24[nIndex] = dy0 * dx;
            //hessian25[nIndex] = dy0 * dy;
            tmp4 = (FPTFOUR)(dy.x, dy.y, dxy.x, dxy.y) * dy.x; //Slightly less accurate
            hessian22[nIndex] = tmp4.x;
            hessian22[secondBufferOffset + nIndex] = tmp4.y;
            hessian24[nIndex] = tmp4.z;
            hessian24[secondBufferOffset + nIndex] = tmp4.w;
            
            //hessian33[nIndex] = pown(dy1,2);
            //hessian34[nIndex] = dy1 * dx;
            //hessian35[nIndex] = dy1 * dy;
            tmp3 = (FPTTHREE)(dy.y, dxy.x, dxy.y) * dy.y; //Slightly less accurate
            hessian33[nIndex] = tmp3.x;
            hessian33[secondBufferOffset + nIndex] = tmp3.y;
            hessian35[nIndex] = tmp3.z;
            
            //hessian44[nIndex] = pown(dx,2);
            //hessian45[nIndex] = dx * dy;
            //hessian55[nIndex] = pown(dy,2);
            tmp3 = (FPTTHREE)(dxy.x, dxy.y, dxy.y) * (FPTTHREE)(dxy.x, dxy.x, dxy.y); //Slightly less accurate
            hessian35[secondBufferOffset + nIndex] = tmp3.x;
            hessian45[nIndex] = tmp3.y;
            hessian45[secondBufferOffset + nIndex] = tmp3.z;
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
