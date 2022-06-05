#ifndef F1_MATRIX_H
#define F1_MATRIX_H

#include <ostream>
#include "F1/Core.h"

namespace f1
{
    class Matrix
    {
    public:
       Matrix() = default;
       Matrix(uint32_t rows, uint32_t columns);
       ~Matrix() noexcept;

        [[nodiscard]] double& At(int row, int column) noexcept { return m_pElements[m_Columns * row + column]; }
        [[nodiscard]] const double& At(int row, int column) const noexcept { return m_pElements[m_Columns * row + column]; }
        [[nodiscard]] uint32_t GetRows() const noexcept { return m_Rows; }
        [[nodiscard]] uint32_t GetColumns() const noexcept { return m_Columns; }

        void Fill(double n);
        void Randomize(int n);
        int ArgMax();
        [[nodiscard]] Matrix Flatten(int axis) const;

        void DumpToFile(const std::string& path) const;
        static Matrix* FromFile(const std::string& path);
        [[nodiscard]] Matrix* Copy() const;

        friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);

        // Operations
        Matrix* Multiply(Matrix* other);
        Matrix* Multiply(double scalar);
        Matrix* Add(Matrix* other);
        Matrix* Add(double scalar);
        Matrix* Subtract(Matrix* other);
        Matrix* Dot(Matrix* other);
        Matrix* Apply(double (*func)(double), Matrix* other) const;
    private:
        bool CheckDimensions(Matrix* other) const;
    private:
        uint32_t m_Rows;
        uint32_t m_Columns;
        double* m_pElements;

    };

    std::ostream& operator<<(std::ostream& os, const Matrix& matrix);

    extern "C"
    {
        typedef f1::Matrix* MatrixHandle;

        [[nodiscard]] MatrixHandle F1_API MatrixCreate(uint32_t rows, uint32_t columns);
        void F1_API MatrixDestroy(MatrixHandle pMatrix);
    }
}

#endif //F1_MATRIX_H
