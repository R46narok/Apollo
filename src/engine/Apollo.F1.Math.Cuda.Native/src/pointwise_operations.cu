#include "pointwise_operations.cuh"
#include <nvtx3/nvToolsExt.h>

__global__ void pointwise_addition_kernel(double* pFirst, double* pSecond, double* pOutput, int iLength)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    // Make sure we do not go pOutput of bounds
    if (id < iLength)
        pOutput[id] = pFirst[id] + pSecond[id];
}

void pointwise_addition(void* pFirst, void* pSecond, void* pOutput, int iLength)
{
    nvtxRangePush(__FUNCTION__);

    int thr_per_blk = 1024;
    int blk_in_grid = ceil(float(iLength) / thr_per_blk);

    pointwise_addition_kernel<<<thr_per_blk, blk_in_grid>>>((double *) pFirst, (double *) pSecond, (double *) pOutput,
                                                            iLength);
    nvtxRangePop();
}

__global__ void pointwise_subtraction_kernel(double* pFirst, double* pSecond, double* pOutput, int iLength)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    // Make sure we do not go pOutput of bounds
    if (id < iLength)
        pOutput[id] = pFirst[id] - pSecond[id];
}

void pointwise_subtraction(void* pFirst, void* pSecond, void* pOutput, int iLength)
{
    nvtxRangePush(__FUNCTION__);
    int thr_per_blk = 1024;
    int blk_in_grid = ceil(float(iLength) / thr_per_blk);

    pointwise_subtraction_kernel<<<thr_per_blk, blk_in_grid>>>((double *) pFirst, (double *) pSecond, (double *) pOutput,
                                                            iLength);
    nvtxRangePop();
}

__global__ void pointwise_multiplication_kernel(double* pFirst, double* pSecond, double* pOutput, int iLength)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    // Make sure we do not go pOutput of bounds
    if (id < iLength)
        pOutput[id] = pFirst[id] * pSecond[id];
}

void pointwise_multiplication(void* pFirst, void* pSecond, void* pOutput, int iLength)
{
    nvtxRangePush(__FUNCTION__);
    int thr_per_blk = 1024;
    int blk_in_grid = ceil(float(iLength) / thr_per_blk);

    pointwise_multiplication_kernel<<<thr_per_blk, blk_in_grid>>>((double *) pFirst, (double *) pSecond, (double *) pOutput,
                                                               iLength);
    nvtxRangePop();
}

__global__ void pointwise_log_kernel(double* pInput, double* pOutput, int iLength)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < iLength)
        pOutput[id] = log(pInput[id]);
}

void pointwise_log(void* pInput, void* pOutput, int iLength)
{
    nvtxRangePush(__FUNCTION__);
    int thr_per_blk = 1024;
    int blk_in_grid = ceil((float)iLength / thr_per_blk);
    pointwise_log_kernel<<<thr_per_blk, blk_in_grid>>>((double*)pInput, (double*)pOutput, iLength);
    nvtxRangePop();
}

__global__ void pointwise_scaled_subtraction_kernel(double* pFirst, double* pSecond, double* pOutput, int iLength, double scale)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    // Make sure we do not go pOutput of bounds
    if (id < iLength)
        pOutput[id] = pFirst[id] - scale * pSecond[id];
}

void pointwise_scaled_subtraction(void* pFirst, void* pSecond, void* pOutput, int iLength, double scale)
{
    nvtxRangePush(__FUNCTION__);
    int thr_per_blk = 1024;
    int blk_in_grid = ceil((float)iLength / thr_per_blk);
    pointwise_scaled_subtraction_kernel<<<thr_per_blk, blk_in_grid>>>((double*)pFirst, (double*)pSecond, (double*)pOutput,
                                                                      iLength, scale);
    nvtxRangePop();
}
