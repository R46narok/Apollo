using System.Collections;
using System.Threading.Tasks.Dataflow;
using Apollo.F1.Math.Common.Buffers;
using Apollo.F1.Math.Exceptions;

namespace Apollo.F1.Math.Common.LinearAlgebra;

public class Matrix : IEnumerable<double>
{
   public static IBufferAllocator BufferFactory { get; set; }
   public static IMatrixOperations Operations { get; set; }

   public IBuffer Buffer { get; }

   public int Rows { get; private set; }
   public int Columns { get; private set; }

   public Matrix(int rows, int columns)
   {
      Rows = rows;
      Columns = columns;

      Buffer = BufferFactory!.Allocate(new BufferDescriptor
      {
         ByteWidth = rows * columns * sizeof(double)
      });
   }

   public Matrix(IBuffer buffer, int rows, int columns)
   {
      Rows = rows;
      Columns = columns;
      Buffer = buffer;
   }

   public Matrix Add(Matrix other)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, other);
      return Operations.Add(this, other);
   }

   public void Add(Matrix other, Matrix output)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, other, output);
      Operations.Add(this, other, output);
   }

   public Matrix Subtract(Matrix other)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, other);
      return Operations.Subtract(this, other);
   }

   public void Subtract(Matrix other, Matrix output)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, other, output);
      Operations.Subtract(this, other, output);
   }

   public void Subtract(Matrix other, Matrix output, double scale)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, other, output);
      Operations.Subtract(this, other, output, scale);
   }

   public double Sum()
   {
      return Operations.Sum(this);
   }

   public Matrix PointwiseMultiply(Matrix other)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, other); 
      return Operations.PointwiseMultiply(this, other);
   }

   public void PointwiseMultiply(Matrix other, Matrix output)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, other, output);
      Operations.PointwiseMultiply(this, other, output);
   }

   public Matrix Multiply(Matrix other)
   {
      if (this.Columns != other.Rows)
             throw new ArgumentException();
      return Operations.Multiply(this, other);
   }

   public void Multiply(Matrix other, Matrix result)
   {
      if (this.Columns != other.Rows)
         throw new ArgumentException();
      Operations.Multiply(this, other, result);
   }

   public void Multiply(double scalar)
   {
      Operations.Multiply(scalar, this, this);
   }

   public void Multiply(double scalar, Matrix output)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, output);
      Operations.Multiply(scalar, this, output);
   }

   public void PointwiseLog(Matrix output)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, output);
      Operations.PointwiseLog(this, output);
   }

   public Matrix PointwiseLog() => Operations.PointwiseLog(this);

   public Matrix Add(double scalar) => Operations.Add(this, scalar);

   public void Add(double scalar, Matrix output)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, output);
      Operations.Add(this, output, scalar);
   }

   public void ApplySigmoid(Matrix output)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, output);
      Operations.ApplySigmoid(this, output);
   }

   public void ApplySigmoidGradient(Matrix output)
   {
      DimensionsMismatchException.ThrowIfNotEqual(this, output);
      Operations.ApplySigmoidGradient(this, output);
   }

   public void InsertColumn(double value, Matrix output)
   {
      if (this.Rows != output.Rows || this.Columns + 1 != output.Columns)
         throw new ArgumentException();
      Operations.InsertColumn(this, output, value);
   }

   public Matrix InsertColumn(double value)
   {
      return Operations.InsertColumn(this, value);
   }

   public void InsertRow(double value, Matrix output)
   {
      if (this.Rows != output.Rows + 1 || this.Columns != output.Columns)
         throw new ArgumentException();
      Operations.InsertRow(this, output, value);
   }

   public Matrix InsertRow(double value) => Operations.InsertRow(this, value);

   public void RemoveColumn(Matrix output)
   {
      if (this.Rows != output.Rows || this.Columns - 1 != output.Columns)
         throw new ArgumentException();
      Operations.RemoveColumn(this, output);
   }

   public Matrix RemoveColumn() => Operations.RemoveColumn(this);

   public void Transpose(Matrix output)
   {
      if (this.Rows != output.Columns || this.Columns != output.Rows)
         throw new ArgumentException();
      Operations.Transpose(this, output);
   }

   public Matrix Transpose() => Operations.Transpose(this);
   public IEnumerator<double> GetEnumerator()
   {
      var data = Buffer.Read();
      foreach (var d in data)
      {
         yield return d;
      }
   }

   IEnumerator IEnumerable.GetEnumerator()
   {
      return GetEnumerator();
   }
}