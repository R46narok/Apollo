using Apollo.F1.Compute.Common.LinearAlgebra;

namespace Apollo.F1.Compute.Common.Interfaces;

public interface IOptimizationProcedure
{
   public void Optimize(ICostFunction function, MatrixStorage[] parameters, 
      MatrixStorage x, MatrixStorage y);
}