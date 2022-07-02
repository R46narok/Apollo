//
// Created by Acer on 2.7.2022 г..
//

#include "memory.cuh"

void* allocate_vram(int bytes)
{
    void* ptr;
    cudaMalloc(&ptr, bytes);
    return ptr;
}

void destroy_vram(void* ptr)
{
    if (ptr != nullptr)
        cudaFree(ptr);
}

void copy_host_to_device(void* src, void* dst, int length)
{
    cudaMemcpy(dst, src, length, cudaMemcpyHostToDevice);
}

void copy_device_to_host(void* src, void* dst, int length)
{
    cudaMemcpy(dst, src, length, cudaMemcpyDeviceToHost);
}