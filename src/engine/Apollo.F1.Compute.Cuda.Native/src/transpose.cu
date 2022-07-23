//
// Created by Acer on 2.7.2022 г..
//

#include "transpose.cuh"

#define BLOCK_DIM 8

__global__ void transpose_kernel(double *odata, double* idata, int width, int height)
{
    unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

    if (xIndex < width && yIndex < height)
    {
        unsigned int index_out  = xIndex + width * yIndex;
        unsigned int index_in = yIndex + height * xIndex;
        odata[index_out] = idata[index_in];
    }
}

void transpose(void* input, void* output, int rows, int columns)
{
    dim3 grid(ceil((double)rows / BLOCK_DIM), ceil((double)columns / BLOCK_DIM), 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);

    transpose_kernel<<< grid, threads >>>((double*)output, (double*)input, rows, columns);
    F1_CUDA_ASSERT(cudaPeekAtLastError());
}