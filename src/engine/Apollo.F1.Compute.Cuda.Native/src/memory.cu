#include "memory.cuh"
#include "stdio.h"

void* allocate_global_memory(int iBytes)
{
   // printf("Allocating %d bytes\n", iBytes);
    void* ptr;
    cudaMalloc(&ptr, iBytes);
    return ptr;
}

void destroy_global_memory(void* ptr)
{
    //printf("Deallocating bytes\n");
    cudaFree(ptr);
}

void copy_host_to_device(void* pSrc, void* pDst, int iLength)
{
    cudaMemcpy(pDst, pSrc, iLength, cudaMemcpyHostToDevice);
}

void copy_device_to_host(void* pSrc, void* pDst, int iLength)
{
    cudaMemcpy(pDst, pSrc, iLength, cudaMemcpyDeviceToHost);
}

void copy_device_to_device(void* pSrc, void* pDst, int iLength)
{
    cudaMemcpy(pDst, pSrc, iLength, cudaMemcpyDeviceToDevice);
}

void device_memset(void* pDst, int iLength, int value)
{
    cudaMemset(pDst, value, iLength);
}
