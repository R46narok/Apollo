#include "scalar.cuh"
#include "nvtx3/nvToolsExt.h"

__global__ void add_scalar_kernel(double* pInput, double* pOutput, int iLength, double scalar)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < iLength)
        pOutput[id] = pInput[id] + scalar;
}

void add_scalar(void* pInput, void* pOutput, int iLength, double scalar)
{
    nvtxRangePush(__FUNCTION__);

    int thr_per_blk = 1024;
    int blk_in_grid = ceil(float(iLength) / thr_per_blk);
    add_scalar_kernel<<<thr_per_blk, blk_in_grid>>>((double*)pInput, (double*)pOutput, iLength, scalar);

    nvtxRangePop();
}

__global__ void subtract_scalar_kernel(double* pInput, double* pOutput, int iLength, double scalar)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < iLength)
        pOutput[id] = pInput[id] - scalar;
}

void subtract_scalar(void* pInput, void* pOutput, int iLength, double scalar)
{
    nvtxRangePush(__FUNCTION__);
    int thr_per_blk = 1024;
    int blk_in_grid = ceil(float(iLength) / thr_per_blk);
    subtract_scalar_kernel<<<thr_per_blk, blk_in_grid>>>((double*)pInput, (double*)pOutput, iLength, scalar);
    nvtxRangePop();
}
