//
// Created by Acer on 2.7.2022 г..
//

#ifndef APOLLO_F1_MATH_CUDA_NATIVE_MULTIPLICATION_CUH
#define APOLLO_F1_MATH_CUDA_NATIVE_MULTIPLICATION_CUH

extern "C"
{
    __declspec(dllexport) void __cdecl multiply(void* first, void* second, void* output, int m, int n, int k);
}

#endif //APOLLO_F1_MATH_CUDA_NATIVE_MULTIPLICATION_CUH
