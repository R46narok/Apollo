using System.Runtime.InteropServices.ComTypes;
using MathNet.Numerics.LinearAlgebra;

namespace Apollo.F1.Math.Extensions;

public static class MatrixExtensions
{ 
   public static void Apply(this Matrix<double> matrix, Func<double, double> function)
   {
      for (int i = 0; i < matrix.RowCount; ++i)
      for (int j = 0; j < matrix.ColumnCount; ++j)
         matrix[i, j] = function(matrix[i, j]);
   }

   public static Matrix<double> ElementwiseMultiplication(this Matrix<double> matrix, Matrix<double> other)
   {
      if (matrix.RowCount != other.RowCount || matrix.ColumnCount != other.ColumnCount)
         throw new ArgumentException("Dimensions dont match");

      int rows = matrix.RowCount;
      int cols = matrix.ColumnCount;
      
      var M = Matrix<double>.Build;
      var result = M.Dense(rows, cols);
      
      for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
         result[i, j] = matrix[i, j] * other[i, j];

      return result;
   }

   public static Matrix<double> ElementwiseMultiplication(this Matrix<double> matrix, Vector<double> other)
   {
      int length = other.Count;

      var M = Matrix<double>.Build;
      var result = M.Dense(1, length);

      for (int i = 0; i < length; ++i)
         result[0, i] = matrix[0, i]* other[i];

      return result;
   }
}