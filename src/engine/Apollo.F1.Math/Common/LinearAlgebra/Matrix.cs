using System.Collections;
using System.Threading.Tasks.Dataflow;
using Apollo.F1.Math.Common.Buffers;

namespace Apollo.F1.Math.Common.LinearAlgebra;

public class Matrix : IEnumerable<double>
{
   public static IBufferFactory BufferFactory { get; set; }
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

   public Matrix Add(Matrix other) => Operations.Add(this, other);
   public void Add(Matrix other, Matrix result) => Operations.Add(this, other, result);

   public Matrix Subtract(Matrix other) => Operations.Subtract(this, other);
   public void Subtract(Matrix other, Matrix result) => Operations.Subtract(this, other, result);

   public Matrix PointwiseMultiply(Matrix other) => Operations.PointwiseMultiply(this, other);
   public void PointwiseMultiply(Matrix other, Matrix output) => Operations.PointwiseMultiply(this, other, output);

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

   public void Multiply(double scalar) => Operations.Multiply(scalar, this, this);
   public void Multiply(double scalar, Matrix output) => Operations.Multiply(scalar, this, output);
   public void PointwiseLog() => Operations.PointwiseLog(this);

   public void ApplySigmoid() => Operations.ApplySigmoid(this);
   public void ApplySigmoidGradient() => Operations.ApplySigmoidGradient(this);

   public void InsertColumn(double value, Matrix output) => Operations.InsertColumn(this, output, value);
   public Matrix InsertColumn(double value) => Operations.InsertColumn(this, value);

   public void InsertRow(double value, Matrix output) => Operations.InsertRow(this, output, value);
   public Matrix InsertRow(double value) => Operations.InsertRow(this, value);

   public void RemoveColumn(Matrix output) => Operations.RemoveColumn(this, output); 
   public Matrix RemoveColumn() => Operations.RemoveColumn(this); 
   
   public void Transpose(Matrix output) => Operations.Transpose(this, output);
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