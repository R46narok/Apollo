using Apollo.F1.Compute.Common.LinearAlgebra;

namespace Apollo.F1.Compute.Common.Interfaces;

public interface ICostFunction
{
   public double ComputeCost(Matrix x, Matrix y);
   public Matrix[] ComputeDerivatives(Matrix x, Matrix y);
}