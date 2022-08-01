using Apollo.F1.Compute.Common.LinearAlgebra;

namespace Apollo.F1.Compute.Optimization.Interfaces;

public interface IOptimizationContext
{
     public void AllocateMemoryForTrainingSet(MatrixStorage[] parameters, int samples);
}