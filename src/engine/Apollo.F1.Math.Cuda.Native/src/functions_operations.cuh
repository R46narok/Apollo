//
// Created by Acer on 2.7.2022 г..
//

#ifndef APOLLO_F1_MATH_CUDA_NATIVE_FUNCTIONS_OPERATIONS_CUH
#define APOLLO_F1_MATH_CUDA_NATIVE_FUNCTIONS_OPERATIONS_CUH

extern "C"
{
    __declspec(dllexport) void __cdecl function_sigmoid(void* ptr, int length);
    __declspec(dllexport) void __cdecl function_sigmoid_gradient(void* ptr, int length);
}

#endif //APOLLO_F1_MATH_CUDA_NATIVE_FUNCTIONS_OPERATIONS_CUH
