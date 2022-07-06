#include "transformations.cuh"

__global__ void insert_column_kernel(double *pOutput, double* pInput, int width, int height, double value)
{
    int idx = threadIdx.x;

    for (int i = 0; i < height; ++i)
    {
        int index_in = idx * height + i;
        int index_out = idx * (height + 1) + (i + 1);
        pOutput[index_out] = pInput[index_in];
    }

    pOutput[idx * (height + 1) + 0] = value;
}

void insert_column(void* pInput, void* pOutput, int iRows, int iColumns, double value)
{
    dim3 grid(1, 1, 1);
    dim3 threads(iRows, 1, 1);

    insert_column_kernel<<< grid, threads >>>((double*)pOutput, (double*)pInput, iRows, iColumns, value);
}

__global__ void insert_row_kernel(double *pOutput, double* pInput, int width, int height, double value)
{
    int idx = threadIdx.x;

    for (int i = 0; i < width; ++i)
    {
        int index_in = i * height + idx;
        int index_out = (i + 1) * height + idx;
        pOutput[index_out] = pInput[index_in];
    }

    pOutput[0 * height + idx] = value;
}

void insert_row(void* pInput, void* pOutput, int iRows, int iColumns, double value)
{
    dim3 grid(1, 1, 1);
    dim3 threads(iColumns, 1, 1);

    insert_row_kernel<<< grid, threads >>>((double*)pOutput, (double*)pInput, iRows, iColumns, value);
}

__global__ void remove_column_kernel(double* pInput, double* pOutput, int width, int height)
{
    int idx = threadIdx.x;
    for (int i = 1; i < height; ++i)
    {
        int index_in = idx * height + i;
        int index_out = idx * (height - 1) + (i - 1);
        pOutput[index_out] = pInput[index_in];
    }
}

void remove_column(void* src, void* dst, int iRows, int iColumns)
{
    dim3 grid(1, 1, 1);
    dim3 threads(iRows, 1, 1);

    remove_column_kernel<<<grid, threads>>>((double*) src,(double*)  dst, iRows, iColumns);
}
