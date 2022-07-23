using Apollo.F1.Compute.Common.Interfaces;
using Apollo.F1.Compute.Common.LinearAlgebra;
using Apollo.F1.Compute.Optimization.Interfaces;

namespace Apollo.F1.Compute.Common.Functions;

public interface ICostFunction<T, U> 
   where T : IOptimizationContext, new()
   where U : IPredictionContext, new()
{
   public MatrixStorage[] Parameters { get; set; }
   public MatrixStorage[] ParametersTransposed { get; set; }
   
   public double ComputeCost(MatrixStorage x, MatrixStorage y, U predictionContext);
   public MatrixStorage[] ComputeDerivatives(MatrixStorage x, MatrixStorage y, T optimizationContext, U predictionContext);
}