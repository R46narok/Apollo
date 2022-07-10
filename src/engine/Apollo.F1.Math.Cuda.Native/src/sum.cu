#include "sum.cuh"
#include "sm_60_atomic_functions.h"
#include <nvtx3/nvToolsExt.h>

__global__ void sum_kernel(double* pInput, double* pOutput, int iRows, int iColumns)
{
    int id = threadIdx.x;

    double sum = 0.0;
    for (int i = 0; i < iRows; ++i)
    {
        for (int j = 0; j < iColumns; ++j)
            sum += pInput[i * iColumns + j];
    }

    *pOutput = sum;
}

void sum(void* pInput, void* pOutput, int iRows, int iColumns)
{
    nvtxRangePush(__FUNCTION__);

    dim3 grid(1, 1, 1);
    dim3 threads(1, 1, 1);

    sum_kernel<<<grid, threads>>>((double*)pInput, (double*)pOutput, iRows, iColumns);

    nvtxRangePop();
}
