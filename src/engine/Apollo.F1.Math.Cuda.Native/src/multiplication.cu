//
// Created by Acer on 2.7.2022 г..
//

#include "multiplication.cuh"

#define BLOCK_SIZE 8

__global__ void multiply_kernel(double* first, double* second, double* output, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    if (col < k && row < m)
    {
        for (int i = 0; i < n; ++i)
        {
            sum += first[row * n + i] * second[i * k + col];
        }
        output[row * k + col] = sum;
    }
}

void multiply(void* first, void* second, void* output, int m, int n, int k)
{
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    multiply_kernel<<<dimGrid, dimBlock>>>((double*)first, (double*)second, (double*)output, m, n, k);
}

__global__ void multiply_scalar_kernel(double* odata, double* idata, int length, double scalar)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < length)
        odata[id] = idata[id] * scalar;
}

void multiply_scalar(void* input, void* output, int length, double scalar)
{
    int thr_per_blk = 256;
    int blk_in_grid = ceil(float(length) / thr_per_blk);

    multiply_scalar_kernel<<<thr_per_blk, blk_in_grid>>>((double *) output, (double *) input, length, scalar);
}
