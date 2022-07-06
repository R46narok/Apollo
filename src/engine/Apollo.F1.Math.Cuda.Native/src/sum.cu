//
// Created by Acer on 6.7.2022 г..
//

#include "sum.cuh"
#include "sm_60_atomic_functions.h"

__global__ void sum_kernel(double* idata, double* odata, int rows, int columns)
{
    int id = threadIdx.x;

    double sum = 0.0;
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
            sum += idata[i * columns + j];
    }

    *odata = sum;
}

void sum(void* input, void* output, int rows, int columns)
{
    dim3 grid(1, 1, 1);
    dim3 threads(1, 1, 1);

    sum_kernel<<<grid, threads>>>((double*)input, (double*)output, rows, columns);
}
