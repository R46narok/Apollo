using Apollo.F1.Compute.Common.Buffers;
using Apollo.F1.Compute.Common.Interfaces;
using Apollo.F1.Compute.Common.LinearAlgebra;
using Apollo.F1.Compute.Helpers;
using Apollo.F1.Compute.Optimization.Interfaces;

namespace Apollo.F1.Compute.Learning.Neural;

public class NeuralPredictionContext : IPredictionContext
{
    private BufferBatch _preactivationBatch;
    private BufferBatch _preactivationGradientBatch;
    private BufferBatch _preactivationGradientBiasedBatch;
    private BufferBatch _activationBatch;
    private BufferBatch _invertedOutputBatch;
    
    private MatrixStorage[] _preactivation;
    private MatrixStorage[] _preactivationGradient;
    private MatrixStorage[] _preactivationGradientBiased;
    private MatrixStorage[] _activation;
    private MatrixStorage[] _invertedOutput;
    
    public MatrixStorage[] Preactivation => _preactivation;
    public MatrixStorage[] Activation => _activation;
    public MatrixStorage[] PreactivationGradient => _preactivationGradient;
    public MatrixStorage[] PreactivationGradientBiased => _preactivationGradientBiased;
    public MatrixStorage[] InvertedOutput => _invertedOutput;
    
    public void AllocateMemoryForPredictionBatch(MatrixStorage[] weights, int batchSize)
    {
       int layers = weights.Length + 1;
       
        BatchMatrixHelper.InitializeBatchAsMatrixArray(out _preactivationBatch, out _preactivation, 
            layers - 1, i => batchSize, i => weights[i].Rows, 
            BufferDataType.Double, "preactivation");
               
        BatchMatrixHelper.InitializeBatchAsMatrixArray(out _preactivationGradientBatch, out _preactivationGradient, 
            layers - 1, i => batchSize, i => weights[i].Rows, 
            BufferDataType.Double, "preactivation");
        
        BatchMatrixHelper.InitializeBatchAsMatrixArray(out _preactivationGradientBiasedBatch, out _preactivationGradientBiased, 
            layers - 1, i => batchSize, i => weights[i].Rows + 1, 
            BufferDataType.Double, "preactivation");
        
        BatchMatrixHelper.InitializeBatchAsMatrixArray(out _activationBatch, out _activation, 
            layers - 1, i => batchSize, i=> weights[i].Rows + 1, 
            BufferDataType.Double, "activation");
        
        BatchMatrixHelper.InitializeBatchAsMatrixArray(out _invertedOutputBatch, out _invertedOutput, 
              2, i => batchSize, i=> weights[1].Rows, 
              BufferDataType.Double, "errorsTransposed");
    }
}