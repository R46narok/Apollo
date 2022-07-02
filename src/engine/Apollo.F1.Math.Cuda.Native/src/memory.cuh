//
// Created by Acer on 2.7.2022 г..
//

#ifndef APOLLO_F1_MATH_CUDA_NATIVE_MEMORY_CUH
#define APOLLO_F1_MATH_CUDA_NATIVE_MEMORY_CUH

extern "C"
{
     __declspec(dllexport) void* __cdecl allocate_vram(int bytes);
     __declspec(dllexport) void __cdecl destroy_vram(void* ptr);

     __declspec(dllexport) void __cdecl copy_host_to_device(void* src, void* dst, int length);
     __declspec(dllexport) void __cdecl copy_device_to_host(void* src, void* dst, int length);
};

#endif //APOLLO_F1_MATH_CUDA_NATIVE_MEMORY_CUH
