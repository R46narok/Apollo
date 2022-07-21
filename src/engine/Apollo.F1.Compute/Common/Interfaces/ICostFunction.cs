using Apollo.F1.Compute.Common.LinearAlgebra;

namespace Apollo.F1.Compute.Common.Interfaces;

public interface ICostFunction
{
   public double ComputeCost(MatrixStorage x, MatrixStorage y);
   public MatrixStorage[] ComputeDerivatives(MatrixStorage x, MatrixStorage y);
}