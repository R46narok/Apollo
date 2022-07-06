//
// Created by Acer on 6.7.2022 г..
//

#include "scalar.cuh"

__global__ void add_scalar_kernel(double* idata, double* odata, int length, double scalar)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < length)
        odata[id] = idata[id] + scalar;
}

void add_scalar(void* input, void* output, int length, double scalar)
{
    int thr_per_blk = 256;
    int blk_in_grid = ceil(float(length) / thr_per_blk);
    add_scalar_kernel<<<thr_per_blk, blk_in_grid>>>((double*)input, (double*)output, length, scalar);
}

__global__ void subtract_scalar_kernel(double* idata, double* odata, int length, double scalar)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < length)
        odata[id] = idata[id] - scalar;
}

void subtract_scalar(void* input, void* output, int length, double scalar)
{
    int thr_per_blk = 256;
    int blk_in_grid = ceil(float(length) / thr_per_blk);
    subtract_scalar_kernel<<<thr_per_blk, blk_in_grid>>>((double*)input, (double*)output, length, scalar);
}
