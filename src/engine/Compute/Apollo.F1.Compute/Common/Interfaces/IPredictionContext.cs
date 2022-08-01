using Apollo.F1.Compute.Common.LinearAlgebra;

namespace Apollo.F1.Compute.Common.Interfaces;

public interface IPredictionContext
{
    public void AllocateMemoryForPredictionBatch(MatrixStorage[] parameters, int batchSize);
}