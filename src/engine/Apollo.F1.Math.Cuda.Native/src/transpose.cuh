//
// Created by Acer on 2.7.2022 г..
//

#ifndef APOLLO_F1_MATH_CUDA_NATIVE_TRANSPOSE_CUH
#define APOLLO_F1_MATH_CUDA_NATIVE_TRANSPOSE_CUH

extern "C"
{
    __declspec(dllexport) void __cdecl transpose(void* input, void* output, int rows, int columns);
};

#endif //APOLLO_F1_MATH_CUDA_NATIVE_TRANSPOSE_CUH
