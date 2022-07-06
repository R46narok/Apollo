//
// Created by Acer on 3.7.2022 г..
//

#ifndef APOLLO_F1_MATH_CUDA_NATIVE_TRANSFORMATIONS_CUH
#define APOLLO_F1_MATH_CUDA_NATIVE_TRANSFORMATIONS_CUH

extern "C"
{
    __declspec(dllexport) void __cdecl insert_column(void* src, void* dst, int rows, int columns, double value);
    __declspec(dllexport) void __cdecl insert_row(void* src, void* dst, int rows, int columns, double value);

    __declspec(dllexport) void __cdecl remove_column(void* src, void* dst, int rows, int columns);
}
#endif //APOLLO_F1_MATH_CUDA_NATIVE_TRANSFORMATIONS_CUH
