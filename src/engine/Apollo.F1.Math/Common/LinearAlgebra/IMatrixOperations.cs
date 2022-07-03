namespace Apollo.F1.Math.Common.LinearAlgebra;

public interface IMatrixOperations
{
   public Matrix Add(Matrix first, Matrix second);
   public void Add(Matrix first, Matrix second, Matrix output);

   public Matrix Subtract(Matrix first, Matrix second);
   public void Subtract(Matrix first, Matrix second, Matrix output);

   public Matrix PointwiseMultiply(Matrix first, Matrix second);
   public void PointwiseMultiply(Matrix first, Matrix second, Matrix output);

   public Matrix Transpose(Matrix matrix);
   public void Transpose(Matrix matrix, Matrix output);
   
   public Matrix Multiply(Matrix first, Matrix second);
   public void Multiply(Matrix first, Matrix second, Matrix output);
}