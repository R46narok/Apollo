#include <iostream>
#include "F1/Math/Matrix.h"

int main()
{
    auto matrix = f1::MatrixCreate(2, 3);
    matrix->At(0, 0) = 3;
    matrix->At(0, 1) = 2;
    matrix->At(0, 2) = 4;
    matrix->At(1, 0) = 9;
    matrix->At(1, 1) = 7;
    matrix->At(1, 2) = 6;

    auto mat2 = f1::MatrixCreate(3, 2);
    mat2->At(0, 0) = 1;
    mat2->At(0, 1) = 5;
    mat2->At(1, 0) = 3;
    mat2->At(1, 1) = 9;
    mat2->At(2, 0) = 7;
    mat2->At(2, 1) = 4;

    f1::MatrixDestroy(matrix);
    f1::MatrixDestroy(mat2);
}