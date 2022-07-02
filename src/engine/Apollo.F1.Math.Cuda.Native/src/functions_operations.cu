//
// Created by Acer on 2.7.2022 г..
//

#include "functions_operations.cuh"
#include "math.h"

__global__ void function_sigmoid_kernel(double* ptr, int length)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < length)
        ptr[id] = 1.0 / (1 + exp(-1.0 * ptr[id]));
}

void function_sigmoid(void* ptr, int length)
{
    int thr_per_blk = 256;
    int blk_in_grid = ceil(float(length) / thr_per_blk);

    function_sigmoid_kernel<<<thr_per_blk, blk_in_grid>>>((double*)ptr, length);
}

__global__ void function_sigmoid_gradient_kernel(double* ptr, int length)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < length)
    {
        double sigmoid = 1.0 / (1 + exp(-1.0 * ptr[id]));
        ptr[id] = sigmoid * (1 - sigmoid);
    }
}

void function_sigmoid_gradient(void* ptr, int length)
{
    int thr_per_blk = 256;
    int blk_in_grid = ceil(float(length) / thr_per_blk);

    function_sigmoid_kernel<<<thr_per_blk, blk_in_grid>>>((double*)ptr, length);
}
