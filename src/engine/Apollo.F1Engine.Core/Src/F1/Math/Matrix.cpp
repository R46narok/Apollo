#include <sstream>
#include <fstream>
#include <filesystem>
#include "Matrix.h"
#include <random>
#include <iostream>
#include <cstring>

namespace f1
{
    Matrix::Matrix(uint32_t rows, uint32_t columns)
    :   m_Rows(rows),
        m_Columns(columns),
        m_pElements(new double[rows * columns])
    {

    }

    Matrix::~Matrix() noexcept
    {
        delete[] m_pElements;
    }

    std::ostream& operator<<(std::ostream& os, const Matrix& matrix)
    {
        uint32_t rows = matrix.GetRows();
        uint32_t columns = matrix.GetColumns();

        os << "Dimensions [" << rows << "x" << columns << "]";
        for (int i = 0; i < rows; ++i)
        {
            os << "\n";
            os << "|";
            for (int j = 0; j < columns; ++j)
            {
                if (j > 0) os << " ";
                os << matrix.At(i, j);
                if (j == matrix.GetColumns() - 1) os << "|";
            }
        }
        return os;
    }

    void Matrix::Fill(double n)
    {
        for (int i = 0; i < m_Rows; ++i)
        {
            for (int j = 0; j < m_Columns; ++j)
            {
                this->At(i, j) = n;
            }
        }
    }

    double UniformDistribution(double low, double high)
    {
        double difference = high - low;
        constexpr int scale = 10000;
        int scaledDifference = (int)(difference * scale);
        return low + (1.0 * (rand() % scaledDifference) / scale);
    }

    void Matrix::Randomize(int n)
    {
        double min = -1.0 / sqrt(n);
        double max = 1.0 / sqrt(n);

        for (int i = 0; i < m_Rows; ++i)
        {
            for (int j = 0; j < m_Columns; ++j)
            {
                this->At(i, j) = UniformDistribution(min, max);
            }
        }
    }

    int Matrix::ArgMax()
    {
        double maxScore = 0;
        int maxIdx = 0;

        for (int i = 0; i < m_Rows; ++i)
        {
            if (this->At(i, 0) > maxScore)
            {
                maxScore = this->At(i, 0);
                maxIdx = i;
            }
        }

        return maxIdx;
    }

    Matrix Matrix::Flatten(int axis) const
    {
        Matrix matrix{};

        if (axis == 0) // column vec
            matrix = Matrix(m_Rows * m_Columns, 1);
        else
            matrix = Matrix(1, m_Rows * m_Columns);

        for (int i = 0; i < m_Rows; ++i)
        {
            for (int j = 0; j < m_Columns; ++j)
            {
                if (axis == 0) matrix.At(i * m_Columns + j, 0) = this->At(i, j);
                else matrix.At(0, i * m_Columns + j) = this->At(i, j);
            }
        }

        return matrix;
    }

    Matrix* Matrix::FromFile(const std::string &path)
    {
        Matrix *pMatrix = nullptr;

        if (std::filesystem::exists(path))
        {
            std::ifstream file(path, std::ios::binary);
            uint32_t rows, columns;

            file.read((char*)&rows, sizeof(uint32_t));
            file.read((char*)&columns, sizeof(uint32_t));

            pMatrix = MatrixCreate(rows, columns);
            for (int i = 0; i < rows * columns; ++i)
            {
                double temp;
                file.read((char*)&temp, sizeof(double));
                pMatrix->m_pElements[i] = temp;
            }

            file.close();
        }

        return pMatrix;
    }

    void Matrix::DumpToFile(const std::string &path) const
    {
        if (!std::filesystem::exists(path))
        {
            std::ofstream file(path, std::ios::binary | std::ios::trunc);
            file.write((char*)this, sizeof(uint32_t) * 2);
            file.write((char*)m_pElements, sizeof(double) * m_Rows * m_Columns);
            file.close();
        }
    }

    Matrix *Matrix::Multiply(Matrix *other)
    {
        Matrix* matrix = nullptr;

        if (CheckDimensions(other))
        {
            matrix = MatrixCreate(m_Rows, other->m_Columns);
            for (int i = 0; i < m_Rows; ++i)
            {
                for (int j = 0; j < m_Columns; ++j)
                {
                    matrix->At(i, j) = this->At(i, j) * other->At(i, j);
                }
            }
        }

        return matrix;
    }

    bool Matrix::CheckDimensions(Matrix *other) const
    {
        return m_Rows == other->m_Rows && m_Columns == other->m_Columns;
    }

    Matrix *Matrix::Multiply(double scalar)
    {
        auto matrix = this->Copy();

        for (int i = 0; i < m_Rows; ++i)
        {
            for (int j = 0; j < m_Columns; ++j)
            {
                matrix->At(i, j) = this->At(i, j) * scalar;
            }
        }

        return matrix;
    }

    Matrix *Matrix::Copy() const
    {
        auto copied = MatrixCreate(m_Rows, m_Columns);
        std::memcpy(copied->m_pElements, m_pElements, sizeof(double) * m_Rows * m_Columns);
        return copied;
    }

    Matrix *Matrix::Add(Matrix *other)
    {
        Matrix* matrix = nullptr;

        if (CheckDimensions(other))
        {
            matrix = MatrixCreate(m_Rows, m_Columns);
            for (int i = 0; i < m_Rows; ++i)
            {
                for (int j = 0; j < m_Columns; ++j)
                {
                    matrix->At(i, j) = this->At(i, j) + other->At(i, j);
                }
            }
        }

        return matrix;
    }

    Matrix *Matrix::Add(double scalar)
    {
        auto matrix = this->Copy();

        for (int i = 0; i < m_Rows; ++i)
        {
            for (int j = 0; j < m_Columns; ++j)
            {
                matrix->At(i, j) = this->At(i, j) + scalar;
            }
        }
        return matrix;
    }

    Matrix *Matrix::Subtract(Matrix *other)
    {
        Matrix* matrix = nullptr;

        if (CheckDimensions(other))
        {
            matrix = MatrixCreate(m_Rows, m_Columns);

            for (int i = 0; i < m_Rows; ++i)
            {
                for (int j = 0; j < m_Columns; ++j)
                {
                    matrix->At(i, j) = this->At(i, j) - other->At(i, j);
                }
            }
        }

        return matrix;
    }

    Matrix *Matrix::Apply(double (*func)(double), Matrix *other) const
    {
        auto matrix = this->Copy();

        for (int i = 0; i < m_Rows; ++i)
        {
            for (int j = 0; j < m_Columns; ++j)
            {
                matrix->At(i, j) = func(matrix->At(i, j));
            }
        }

        return matrix;
    }

    Matrix *Matrix::Dot(Matrix *other)
    {
        Matrix* matrix = nullptr;

        if (m_Rows == other->m_Columns)
        {
            matrix = MatrixCreate(m_Rows, other->m_Columns);
            for (int i = 0; i < m_Rows; ++i)
            {
                for (int j = 0; j < other->m_Columns; ++j)
                {
                    double sum = 0.0;
                    for (int k = 0; k < other->m_Rows; ++k)
                    {
                        sum += this->At(i, k) * other->At(k, j);
                    }
                    matrix->At(i, j) = sum;
                }
            }
        }

        return matrix;
    }

    MatrixHandle MatrixCreate(uint32_t rows, uint32_t columns)
    {
        return new Matrix(rows, columns);
    }

    void MatrixDestroy(MatrixHandle pMatrix)
    {
        delete pMatrix;
    }


} // f1