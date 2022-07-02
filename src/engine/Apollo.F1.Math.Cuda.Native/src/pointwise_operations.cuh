//
// Created by Acer on 2.7.2022 г..
//

#ifndef APOLLO_F1_MATH_CUDA_NATIVE_POINTWISE_OPERATIONS_CUH
#define APOLLO_F1_MATH_CUDA_NATIVE_POINTWISE_OPERATIONS_CUH

extern "C"
{
   __declspec(dllexport) void __cdecl pointwise_addition(void* first, void* second, void* output, int length);
   __declspec(dllexport) void __cdecl pointwise_subtraction(void* first, void* second, void* output, int length);
   __declspec(dllexport) void __cdecl pointwise_multiplication(void* first, void* second, void* output, int length);
};

#endif //APOLLO_F1_MATH_CUDA_NATIVE_POINTWISE_OPERATIONS_CUH
