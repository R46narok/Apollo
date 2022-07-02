using Apollo.F1.Math.Common.Buffers;

namespace Apollo.F1.Math.Common.LinearAlgebra;

public class Matrix 
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

   private Matrix(IBuffer buffer, int rows, int columns)
   {
   }

   public Matrix Add(Matrix other) => Operations.Add(this, other);
   public void Add(Matrix other, Matrix result) => Operations.Add(this, other, result);

   public Matrix Subtract(Matrix other) => Operations.Subtract(this, other);
   public void Subtract(Matrix other, Matrix result) => Operations.Subtract(this, other, result);

   public Matrix PointwiseMultiply(Matrix other) => Operations.PointwiseMultiply(this, other);
   public void PointwiseMultiply(Matrix other, Matrix output) => Operations.PointwiseMultiply(this, other, output);
}