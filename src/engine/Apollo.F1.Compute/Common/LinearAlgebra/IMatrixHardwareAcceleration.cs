namespace Apollo.F1.Compute.Common.LinearAlgebra;

public interface IMatrixHardwareAcceleration
{
   public MatrixStorage Add(MatrixStorage first, MatrixStorage second);
   public void Add(MatrixStorage first, MatrixStorage second, MatrixStorage output);

   public MatrixStorage Subtract(MatrixStorage first, MatrixStorage second);
   public void Subtract(MatrixStorage first, MatrixStorage second, MatrixStorage output);

   public MatrixStorage PointwiseMultiply(MatrixStorage first, MatrixStorage second);
   public void PointwiseMultiply(MatrixStorage first, MatrixStorage second, MatrixStorage output);

   public MatrixStorage Transpose(MatrixStorage matrix);
   public void Transpose(MatrixStorage matrix, MatrixStorage output);
   
   public MatrixStorage Multiply(MatrixStorage first, MatrixStorage second);
   public void Multiply(MatrixStorage first, MatrixStorage second, MatrixStorage output);

   public void PointwiseLog(MatrixStorage matrix, MatrixStorage output);
   public MatrixStorage PointwiseLog(MatrixStorage matrix);

   public MatrixStorage Add(MatrixStorage input, double scalar);
   public void Add(MatrixStorage input, MatrixStorage output, double scalar);

   public double Sum(MatrixStorage matrix);

   public MatrixStorage Subtract(MatrixStorage first, MatrixStorage second, double scale);
   public void Subtract(MatrixStorage first, MatrixStorage second, MatrixStorage output, double scale);
   
   public MatrixStorage Multiply(double scalar);
   public void Multiply(double scalar, MatrixStorage input, MatrixStorage output);
   
   public void ApplySigmoid(MatrixStorage matrix, MatrixStorage output);
   public void ApplySigmoidGradient(MatrixStorage matrix, MatrixStorage output);

   public void InsertColumn(MatrixStorage matrix, MatrixStorage output, double value);
   public MatrixStorage InsertColumn(MatrixStorage matrix, double value);
   
   public void InsertRow(MatrixStorage matrix, MatrixStorage output, double value);
   public MatrixStorage InsertRow(MatrixStorage matrix, double value);

   public void RemoveColumn(MatrixStorage matrix, MatrixStorage output);
   public MatrixStorage RemoveColumn(MatrixStorage matrix);
}