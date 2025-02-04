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

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
//SDT source data type will be the source type passed by the compiler
//TDT target data type will be the target type passed by the compiler
//CMD saturated cast function will be passed by the compiler
//MAXVAL scaling factor will be passed by the compiler
__kernel void Convert(__global const SDT *source, __global TDT *target, const int size)
{
    __private int nIndex = get_global_id(0); // this directly corresponds to the offset
    if(nIndex >= size)
    {
        return;
    }
    target[nIndex] = ((TDT) source[nIndex]) / MAXVAL;
} 
__kernel void deConvert(__global const TDT *source, __global SDT *target, const int size)
{
    __private int nIndex = get_global_id(0); // this directly corresponds to the offset
    if(nIndex >= size)
    {
        return;
    }
    target[nIndex] = CMD (source[nIndex] * MAXVAL) ;
}
