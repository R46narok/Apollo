using Apollo.F1.Math.Common.LinearAlgebra;

namespace Apollo.F1.Math.Common.Interfaces;

public interface IOptimizationProcedure
{
   public void Optimize(ICostFunction function, Matrix[] parameters, Matrix[] derivatives,
      Matrix x, Matrix y);
}