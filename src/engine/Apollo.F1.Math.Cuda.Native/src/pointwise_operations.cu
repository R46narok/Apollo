//
// Created by Acer on 2.7.2022 г..
//

#include "pointwise_operations.cuh"

__global__ void pointwise_addition_kernel(double* first, double* second, double* output, int length)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < length)
        output[id] = first[id] + second[id];
}

void pointwise_addition(void* first, void* second, void* output, int length)
{
    int thr_per_blk = 256;
    int blk_in_grid = ceil(float(length) / thr_per_blk);

    pointwise_addition_kernel<<<thr_per_blk, blk_in_grid>>>((double *) first, (double *) second, (double *) output,
                                                            length);
}

__global__ void pointwise_subtraction_kernel(double* first, double* second, double* output, int length)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < length)
        output[id] = first[id] - second[id];
}

void pointwise_subtraction(void* first, void* second, void* output, int length)
{
    int thr_per_blk = 256;
    int blk_in_grid = ceil(float(length) / thr_per_blk);

    pointwise_subtraction_kernel<<<thr_per_blk, blk_in_grid>>>((double *) first, (double *) second, (double *) output,
                                                            length);
}

__global__ void pointwise_multiplication_kernel(double* first, double* second, double* output, int length)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < length)
        output[id] = first[id] * second[id];
}

void pointwise_multiplication(void* first, void* second, void* output, int length)
{
    int thr_per_blk = 256;
    int blk_in_grid = ceil(float(length) / thr_per_blk);

    pointwise_multiplication_kernel<<<thr_per_blk, blk_in_grid>>>((double *) first, (double *) second, (double *) output,
                                                               length);
}
