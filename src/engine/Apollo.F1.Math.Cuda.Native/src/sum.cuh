//
// Created by Acer on 6.7.2022 г..
//

#ifndef APOLLO_F1_MATH_CUDA_NATIVE_SUM_CUH
#define APOLLO_F1_MATH_CUDA_NATIVE_SUM_CUH

extern "C"
{
    __declspec(dllexport) void __cdecl sum(void* input, void* output, int rows, int columns);
};

#endif //APOLLO_F1_MATH_CUDA_NATIVE_SUM_CUH
