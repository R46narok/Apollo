using Apollo.F1.Math.Common.Buffers;

namespace Apollo.F1.Math.Common.LinearAlgebra;

public class Matrix
{
   private readonly IBuffer _buffer;

   public int Rows { get; private set; }
   public int Columns { get; private set; }
   
   private Matrix(int rows, int columns)
   {
      Rows = rows;
      Columns = columns;
   }
}