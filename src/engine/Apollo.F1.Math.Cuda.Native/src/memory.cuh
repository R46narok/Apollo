//
// Created by Acer on 2.7.2022 г..
//

#ifndef APOLLO_F1_MATH_CUDA_NATIVE_MEMORY_CUH
#define APOLLO_F1_MATH_CUDA_NATIVE_MEMORY_CUH

extern "C"
{
    /// Host instruction to allocate memory on the graphics device (VRAM).
    /// \param bytes Length of the buffer in bytes
    /// \return A pointer to GPU global memory
     __declspec(dllexport) void* __cdecl allocate_vram(int bytes);
     __declspec(dllexport) void __cdecl destroy_vram(void* ptr);

     __declspec(dllexport) void __cdecl copy_host_to_device(void* src, void* dst, int length);
     __declspec(dllexport) void __cdecl copy_device_to_host(void* src, void* dst, int length);
     __declspec(dllexport) void __cdecl copy_device_to_device(void* src, void* dst, int length);

     __declspec(dllexport) void __cdecl device_memset(void* dst, int length, int value);
};

#endif //APOLLO_F1_MATH_CUDA_NATIVE_MEMORY_CUH
