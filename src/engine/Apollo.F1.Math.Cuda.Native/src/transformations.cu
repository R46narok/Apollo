#include "transformations.cuh"

#define BLOCK_SIZE 32

__global__ void insert_column_kernel(double *pOutput, double* pInput, int width, int height, double value)
{
    unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

    if (xIndex < width && yIndex < height)
    {
        int index_in = xIndex * height + yIndex;
        int index_out = xIndex * (height + 1) + (yIndex + 1);

        pOutput[index_out] = pInput[index_in];
    }

    if (xIndex == 0)
        pOutput[xIndex * (height + 1) + 0] = value;
}

void insert_column(void* pInput, void* pOutput, int iRows, int iColumns, double value)
{
    dim3 grid(iRows / BLOCK_SIZE, iColumns / BLOCK_SIZE, 1);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1);

    insert_column_kernel<<< grid, threads >>>((double*)pOutput, (double*)pInput, iRows, iColumns, value);
}

__global__ void insert_row_kernel(double *pOutput, double* pInput, int width, int height, double value)
{
    unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

    if (xIndex < width && yIndex < height)
    {
        int index_in = xIndex * height + yIndex;
        int index_out = (xIndex + 1) * height + yIndex;
        pOutput[index_out] = pInput[index_in];
    }


    pOutput[0 * height + xIndex] = value;
}

void insert_row(void* pInput, void* pOutput, int iRows, int iColumns, double value)
{
    dim3 grid(iRows / BLOCK_SIZE, iColumns / BLOCK_SIZE, 1);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1);

    insert_row_kernel<<< grid, threads >>>((double*)pOutput, (double*)pInput, iRows, iColumns, value);
}

__global__ void remove_column_kernel(double* pInput, double* pOutput, int width, int height)
{
    unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

    if (xIndex < width && yIndex < height)
    {
        int index_in = xIndex * height + yIndex;
        int index_out = xIndex * (height - 1) + (yIndex - 1);
        pOutput[index_out] = pInput[index_in];
    }
}

void remove_column(void* src, void* dst, int iRows, int iColumns)
{
    dim3 grid(iRows / BLOCK_SIZE, iColumns / BLOCK_SIZE, 1);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1);

    remove_column_kernel<<<grid, threads>>>((double*) src,(double*)  dst, iRows, iColumns);
}
