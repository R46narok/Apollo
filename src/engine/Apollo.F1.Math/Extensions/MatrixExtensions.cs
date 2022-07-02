using System.Runtime.InteropServices.ComTypes;
using MathNet.Numerics.LinearAlgebra;

namespace Apollo.F1.Math.Extensions;

public static class MatrixExtensions
{
    public static Matrix<double> InsertBiasColumn(this Matrix<double> matrix)
    {
        int rows = matrix.RowCount;
        const double biasValue = 1.0;

        matrix = matrix.InsertColumn(0,
            Vector<double>.Build.Dense(rows, biasValue));

        return matrix;
    }

    public static Matrix<double> RemoveBiasColumn(this Matrix<double> matrix)
    {
        return matrix.RemoveColumn(0);
    }

    public static Matrix<double> InsertBiasRow(this Matrix<double> matrix)
    {
        int columns = matrix.ColumnCount;
        const double biasValue = 1.0;
        matrix = matrix.InsertRow(0,
                        Vector<double>.Build.Dense(columns, biasValue));
        return matrix;
    }
}