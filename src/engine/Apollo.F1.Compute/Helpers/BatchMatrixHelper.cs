using Apollo.F1.Compute.Common.Buffers;
using Apollo.F1.Compute.Common.LinearAlgebra;

namespace Apollo.F1.Compute.Helpers;

public static class BatchMatrixHelper
{
    public static void InitializeBatchAsMatrixArray(out BufferBatch batch, out MatrixStorage[] matrices,
               int length, Func<int, int> rowFunction, Func<int, int> columnFunction,
               BufferDataType dataType, string name)
    {
        var batchElements = new BufferBatchElement[length];
        for (int i = 0; i < length; ++i)
            batchElements[i] = new BufferBatchElement(sizeof(double) * rowFunction(i) * columnFunction(i), dataType, name);
    
        batch = new BufferBatch(MatrixStorage.BufferFactory, batchElements);
        matrices = new MatrixStorage[length];
    
        for (int i = 0; i < length; ++i)
            matrices[i] = new MatrixStorage(batch[i], rowFunction(i), columnFunction(i));
    }
}