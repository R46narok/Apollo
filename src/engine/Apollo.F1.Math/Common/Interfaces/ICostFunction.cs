using Apollo.F1.Math.Common.LinearAlgebra;

namespace Apollo.F1.Math.Common.Interfaces;

public interface ICostFunction
{
   public double ComputeCost(Matrix x, Matrix y);
   public void ComputeDerivatives(Matrix x, Matrix y);
}