#include "multiplication.cuh"
#include "stdio.h"
#include "nvtx3/nvToolsExt.h"

#define BLOCK_SIZE 16

__global__ void multiply_kernel(const double* pFirst, const double* pSecond, double* pOutput,
                                int firstRows, int firstColumns, int secondColumns)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < firstRows && col < secondColumns)
    {
        double sum = 0.0;
        int i = 0;
        for (i = 0; i < firstColumns; ++i)
        {
            sum += pFirst[row * firstColumns + i] * pSecond[i * secondColumns + col];
        }
        pOutput[row * secondColumns + col] = sum;
    }
}

void multiply(void* pFirst, void* pSecond, void* pOutput,
              int firstRows, int firstColumns, int secondColumns)
{
    nvtxRangePush(__FUNCTION__);

    unsigned int grid_rows = ceil((double)firstRows / BLOCK_SIZE);
    unsigned int grid_cols = ceil((double)secondColumns / BLOCK_SIZE);

    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    multiply_kernel<<<dimGrid, dimBlock>>>((double*)pFirst, (double*)pSecond, (double*)pOutput,
                                           firstRows, firstColumns, secondColumns);

    nvtxRangePop();
}

__global__ void multiply_scalar_kernel(double* pOutput, double* pInput, int iLength, double scalar)
{
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < iLength;
         i += blockDim.x * gridDim.x)
    {
        pOutput[i] = pInput[i] * scalar;
    }
}

void multiply_scalar(void* input, void* pOutput, int iLength, double scalar)
{
    multiply_scalar_kernel<<<512, 256>>>((double *) pOutput, (double *) input, iLength, scalar);
}
