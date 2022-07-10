#include "multiplication.cuh"

#include "nvtx3/nvToolsExt.h"
#define BLOCK_SIZE 8

__global__ void multiply_kernel(double* pFirst, double* pSecond, double* pOutput, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    if (col < k && row < m)
    {
        for (int i = 0; i < n; ++i)
        {
            sum += pFirst[row * n + i] * pSecond[i * k + col];
        }
        pOutput[row * k + col] = sum;
    }
}

void multiply(void* pFirst, void* pSecond, void* pOutput,
              int firstRows, int firstColumns, int secondColumns)
{
    nvtxRangePush(__FUNCTION__);

    unsigned int grid_rows = (firstRows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (secondColumns + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    multiply_kernel<<<dimGrid, dimBlock>>>((double*)pFirst, (double*)pSecond, (double*)pOutput, firstRows, firstColumns, secondColumns);

    nvtxRangePop();
}

__global__ void multiply_scalar_kernel(double* pOutput, double* pInput, int iLength, double scalar)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < iLength)
        pOutput[id] = pInput[id] * scalar;
}

void multiply_scalar(void* input, void* pOutput, int iLength, double scalar)
{
    int thr_per_blk =  1024;
    int blk_in_grid = ceil(float(iLength) / thr_per_blk);

    multiply_scalar_kernel<<<thr_per_blk, blk_in_grid>>>((double *) pOutput, (double *) input, iLength, scalar);
}
