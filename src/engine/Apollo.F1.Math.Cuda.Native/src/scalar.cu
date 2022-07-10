#include "scalar.cuh"
#include "nvtx3/nvToolsExt.h"

__global__ void add_scalar_kernel(double* pInput, double* pOutput, int iLength, double scalar)
{
    for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < iLength;
         i += blockDim.x * gridDim.x)
    {
        pOutput[i] = pInput[i] + scalar;
    }
}

void add_scalar(void* pInput, void* pOutput, int iLength, double scalar)
{
    nvtxRangePush(__FUNCTION__);

    add_scalar_kernel<<<512, 256>>>((double*)pInput, (double*)pOutput, iLength, scalar);

    nvtxRangePop();
}

__global__ void subtract_scalar_kernel(double* pInput, double* pOutput, int iLength, double scalar)
{
    for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < iLength;
         i += blockDim.x * gridDim.x)
    {
       pOutput[i] = pInput[i] - scalar;
    }
}

void subtract_scalar(void* pInput, void* pOutput, int iLength, double scalar)
{
    nvtxRangePush(__FUNCTION__);
    subtract_scalar_kernel<<<512, 256>>>((double*)pInput, (double*)pOutput, iLength, scalar);
    nvtxRangePop();
}
