//
// Created by Acer on 3.7.2022 г..
//

#include "transformations.cuh"

__global__ void insert_column_kernel(double *odata, double* idata, int width, int height, double value)
{
    int idx = threadIdx.x;

    for (int i = 0; i < height; ++i)
    {
        int index_in = idx * height + i;
        int index_out = idx * (height + 1) + (i + 1);
        odata[index_out] = idata[index_in];
    }

    odata[idx * (height + 1) + 0] = value;
}

void insert_column(void* input, void* output, int rows, int columns, double value)
{
    dim3 grid(1, 1, 1);
    dim3 threads(rows, 1, 1);

    insert_column_kernel<<< grid, threads >>>((double*)output, (double*)input, rows, columns, value);
}

__global__ void insert_row_kernel(double *odata, double* idata, int width, int height, double value)
{
    int idx = threadIdx.x;

    for (int i = 0; i < width; ++i)
    {
        int index_in = i * height + idx;
        int index_out = (i + 1) * height + idx;
        odata[index_out] = idata[index_in];
    }

    odata[0 * height + idx] = value;
}

void insert_row(void* input, void* output, int rows, int columns, double value)
{
    dim3 grid(1, 1, 1);
    dim3 threads(columns, 1, 1);

    insert_row_kernel<<< grid, threads >>>((double*)output, (double*)input, rows, columns, value);
}

__global__ void remove_column_kernel(double* idata, double* odata, int width, int height)
{
    int idx = threadIdx.x;
    for (int i = 1; i < height; ++i)
    {
        int index_in = idx * height + i;
        int index_out = idx * (height - 1) + (i - 1);
        odata[index_out] = idata[index_in];
    }
}

void remove_column(void* src, void* dst, int rows, int columns)
{
    dim3 grid(1, 1, 1);
    dim3 threads(rows, 1, 1);

    remove_column_kernel<<<grid, threads>>>((double*) src,(double*)  dst, rows, columns);
}
