using Apollo.F1.Compute.Common.Functions;
using Apollo.F1.Compute.Common.Interfaces;
using Apollo.F1.Compute.Common.LinearAlgebra;

namespace Apollo.F1.Compute.Optimization.Interfaces;

public interface IOptimizationProcedure<T, U>
    where T : IOptimizationContext, new()
    where U : IPredictionContext, new()
{
    public void Optimize(ICostFunction<T, U> function, MatrixStorage x, MatrixStorage y);
}