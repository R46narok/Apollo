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
}