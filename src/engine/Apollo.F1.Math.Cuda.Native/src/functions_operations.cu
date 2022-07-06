#include "functions_operations.cuh"
#include <cmath>

__global__ void function_sigmoid_kernel(double* pElements, int iLength)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < iLength)
        pElements[id] = 1.0 / (1 + exp(-1.0 * pElements[id]));
}

void function_sigmoid(void* pElements, int iLength)
{
    int thr_per_blk = 256;
    int blk_in_grid = ceil(float(iLength) / thr_per_blk);

    function_sigmoid_kernel<<<thr_per_blk, blk_in_grid>>>((double*)pElements, iLength);
}

__global__ void function_sigmoid_gradient_kernel(double* pElements, int iLength)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < iLength)
    {
        double sigmoid = 1.0 / (1 + exp(-1.0 * pElements[id]));
        pElements[id] = sigmoid * (1 - sigmoid);
    }
}

void function_sigmoid_gradient(void* pElements, int iLength)
{
    int thr_per_blk = 256;
    int blk_in_grid = ceil(float(iLength) / thr_per_blk);

    function_sigmoid_gradient_kernel<<<thr_per_blk, blk_in_grid>>>((double*)pElements, iLength);
}
