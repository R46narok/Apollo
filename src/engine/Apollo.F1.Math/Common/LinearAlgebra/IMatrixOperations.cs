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

   public void PointwiseLog(Matrix matrix, Matrix output);
   public Matrix PointwiseLog(Matrix matrix);

   public Matrix Add(Matrix input, double scalar);
   public void Add(Matrix input, Matrix output, double scalar);

   public double Sum(Matrix matrix);

   public Matrix Subtract(Matrix first, Matrix second, double scale);
   public void Subtract(Matrix first, Matrix second, Matrix output, double scale);
   
   public Matrix Multiply(double scalar);
   public void Multiply(double scalar, Matrix input, Matrix output);
   
   public void ApplySigmoid(Matrix matrix);
   public void ApplySigmoidGradient(Matrix matrix);

   public void InsertColumn(Matrix matrix, Matrix output, double value);
   public Matrix InsertColumn(Matrix matrix, double value);
   
   public void InsertRow(Matrix matrix, Matrix output, double value);
   public Matrix InsertRow(Matrix matrix, double value);

   public void RemoveColumn(Matrix matrix, Matrix output);
   public Matrix RemoveColumn(Matrix matrix);
}