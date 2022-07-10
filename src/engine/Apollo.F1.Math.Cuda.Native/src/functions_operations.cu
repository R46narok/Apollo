#include "functions_operations.cuh"
#include "nvtx3/nvToolsExt.h"
#include <cmath>
#include <stdio.h>

__global__ void function_sigmoid_kernel(double* pInput, double* pOutput, int iLength)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < iLength)
        pOutput[id] = 1.0 / (1 + exp(-1.0 * pInput[id]));
}

void function_sigmoid(void* pInput, void* pOutput, int iLength)
{
    nvtxRangePush(__FUNCTION__);
    int thr_per_blk = 1024;
    int blk_in_grid = ceil(float(iLength) / thr_per_blk);

    function_sigmoid_kernel<<<thr_per_blk, blk_in_grid>>>((double*)pInput, (double*)pOutput, iLength);

    nvtxRangePop();
}

__global__ void function_sigmoid_gradient_kernel(double* pInput, double* pOutput, int iLength)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < iLength)
    {
        double sigmoid = 1.0 / (1 + exp(-1.0 * pInput[id]));
        pOutput[id] = sigmoid * (1 - sigmoid);
    }
}

void function_sigmoid_gradient(void* pInput, void* pOutput, int iLength)
{
    nvtxRangePush(__FUNCTION__);
    int thr_per_blk = 1024;
    int blk_in_grid = ceil(float(iLength) / thr_per_blk);

    function_sigmoid_gradient_kernel<<<thr_per_blk, blk_in_grid>>>((double*)pInput, (double*)pOutput, iLength);
    nvtxRangePop();
}
