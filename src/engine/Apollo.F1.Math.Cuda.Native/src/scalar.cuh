//
// Created by Acer on 6.7.2022 г..
//

#ifndef APOLLO_F1_MATH_CUDA_NATIVE_SCALAR_CUH
#define APOLLO_F1_MATH_CUDA_NATIVE_SCALAR_CUH

extern "C"
{
    __declspec(dllexport) void __cdecl add_scalar(void* input, void* output, int length, double scalar);
    __declspec(dllexport) void __cdecl subtract_scalar(void* input, void* output, int length, double scalar);
}

#endif //APOLLO_F1_MATH_CUDA_NATIVE_SCALAR_CUH
