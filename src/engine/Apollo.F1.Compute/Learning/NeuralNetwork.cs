using Apollo.F1.Compute.Common.Interfaces;
using Apollo.F1.Compute.Common.LinearAlgebra;
using Apollo.F1.Compute.Exceptions;
using Apollo.F1.Compute.Neural;
using Apollo.F1.Compute.Common.Buffers;

namespace Apollo.F1.Compute.Learning;

public class NeuralNetwork : ICostFunction
{
    private readonly int[] _layers;
    private readonly double _learningRate;
    private readonly double _regularizationTerm;
    private readonly double _distributionUpperBound;
    private readonly double _distributionLowerBound;
    private readonly IMatrixHardwareAcceleration _matrixOperations;

    private MatrixStorage[] _weights = null!;
    private MatrixStorage[] _tempWeights = null!;
    private MatrixStorage[] _weightsTransposed = null!;
    private MatrixStorage[] _derivatives = null!;
    
    public NeuralNetwork(NeuralNetworkOptions options)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));

        _layers = options.Layers;
        _learningRate = options.LearningRate;
        _regularizationTerm = options.RegularizationTerm;
        _distributionLowerBound = options.DistributionBoundaries.Item1;
        _distributionUpperBound = options.DistributionBoundaries.Item2;
        _matrixOperations = MatrixStorage.Operations;

        OutputLayerIdx = _layers.Length - 1;
        
        ValidateNetworkArchitecture();
        InitializeWeights();
        InitializeDerivatives();
        InitializeBiasNeurons();
    }

    private void ValidateNetworkArchitecture()
    {
        if (_layers.Length < 2) throw new ArchitectureException();
    }

    private void InitializeWeights()
    {
        int length = _layers.Length - 1;
        _weights = new MatrixStorage[length];
        _weightsTransposed = new MatrixStorage[length];
        _tempWeights = new MatrixStorage[length];
        
        for (int i = 0; i < length; ++i)
        {
            var cpuBuffer = new double[_layers[i + 1] * (_layers[i] + 1)];
            for (int j = 0; j < cpuBuffer.Length; ++j)
            {
                var axis = (double)(Random.Shared.Next(0, 2) * 2 - 1);
                var distribution = Random.Shared.NextSingle();
                var value = axis * System.Math.Sqrt(6) * distribution;
                cpuBuffer[j] = value;
            }

            _weights[i] = new MatrixStorage(_layers[i + 1], _layers[i] + 1);
            _weights[i].Buffer.Upload(cpuBuffer);
            _tempWeights[i] = new MatrixStorage(_layers[i + 1], _layers[i] + 1);
            _weightsTransposed[i] = _weights[i].Transpose();
        }
    }

    private void InitializeDerivatives()
    {
        int length = _layers.Length - 1;
        _derivatives = new MatrixStorage[length];

        for (int i = 0; i < length; ++i)
            _derivatives[i] = new MatrixStorage(_layers[i + 1], _layers[i] + 1);
    }
    
    /// <summary>
    /// Adds bias neuron to each layer except the last.
    /// </summary>
    private void InitializeBiasNeurons()
    {
        int l = _layers.Length - 1;
        for (int i = 0; i < l; ++i)
        {
            _layers[i]++;
        }
    }

    private BufferBatch _preactivationBatch;
    private BufferBatch _preactivationGradientBatch;
    private BufferBatch _preactivationGradientBiasedBatch;
    private BufferBatch _activationBatch;
    private BufferBatch _errorBatch;
    private BufferBatch _errorsTransposedBatch;
    private BufferBatch _errorsBiasedBatch;
    private BufferBatch _negativeOutputBatch;
    private MatrixStorage[] _preactivation = null!;
    private MatrixStorage[] _preactivationGradient = null!;
    private MatrixStorage[] _preactivationGradientBiased = null!;
    private MatrixStorage[] _activation;
    private MatrixStorage[] _errors = null!;
    private MatrixStorage[] _errorsTransposed = null!;
    private MatrixStorage[] _errorsBiased = null!;
    private MatrixStorage[] _negativeOutput = null!;

    public void InitializeBufferBatches(int samples)
    {
        InitializeBatchAsMatrixArray(out _preactivationBatch, out _preactivation, 
            _layers.Length - 1, i => samples, i => _weightsTransposed[i].Columns,
            BufferDataType.Double, "preactivation");
        
        InitializeBatchAsMatrixArray(out _preactivationGradientBatch, out _preactivationGradient, 
            _layers.Length - 1, i => samples, i => _weightsTransposed[i].Columns, 
            BufferDataType.Double, "preactivation");

        InitializeBatchAsMatrixArray(out _preactivationGradientBiasedBatch, out _preactivationGradientBiased, 
            _layers.Length - 1, i => samples, i => _weightsTransposed[i].Columns + 1, 
            BufferDataType.Double, "preactivation");
        
        InitializeBatchAsMatrixArray(out _activationBatch, out _activation, 
            _layers.Length - 1, i => samples, i=> _weightsTransposed[i].Columns + 1, 
            BufferDataType.Double, "activation");
        
        InitializeBatchAsMatrixArray(out _errorBatch, out _errors,
            _layers.Length - 1, i => samples, i => _weightsTransposed[i].Columns, 
            BufferDataType.Double, "errors");
        
        InitializeBatchAsMatrixArray(out _errorsTransposedBatch, out _errorsTransposed, 
            _layers.Length - 1, i => _weightsTransposed[i].Columns, i => samples, 
            BufferDataType.Double, "errorsTransposed");
        
        InitializeBatchAsMatrixArray(out _errorsBiasedBatch, out _errorsBiased, 
            _layers.Length - 2, i => samples, i=> _weightsTransposed[i].Columns + 1, 
            BufferDataType.Double, "errorsTransposed");
        
        InitializeBatchAsMatrixArray(out _negativeOutputBatch, out _negativeOutput, 
                    2, i => samples, i=> _weightsTransposed[1].Columns, 
                    BufferDataType.Double, "errorsTransposed");
    }

    private void InitializeBatchAsMatrixArray(out BufferBatch batch, out MatrixStorage[] matrices,
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

    private const int InputLayerIdx = 0;
    private readonly int OutputLayerIdx;

    private bool IsOutputLayer(int idx) => idx == OutputLayerIdx;
    private bool IsInputLayer(int idx) => idx == InputLayerIdx;
    private bool IsHiddenLayer(int idx) => !IsInputLayer(idx) && !IsOutputLayer(idx);
    
    private bool IsNotOutputLayer(int idx) => !IsOutputLayer(idx);
    private bool IsNotInputLayer(int idx) => !IsInputLayer(idx);
    private bool IsNotHiddenLayer(int idx) => !IsHiddenLayer(idx);

    
    /// <summary>
    /// Computes the output of a neural network based on a given input
    /// using the forward propagation algorithm.
    /// </summary>
    /// <param name="x">Matrix for the input layer with already added bias column</param>
    /// <param name="computeGradients"></param>
    /// <returns>[training samples x number of output units] matrix, which contains the predictions</returns>
    public MatrixStorage FeedForward(MatrixStorage x, bool computeGradients = false)
    {
        var context = MatrixComputeContext.Create(MatrixStorage.Operations);
        
        var last = x;

        var length = _layers.Length - 1; // excluding the first(input) layer
        for (int i = 0; i < length; ++i)
        {
            context.PerformOn(last).And(_weightsTransposed[i]).MultiplyInto(_preactivation[i]); // preactivation of the current layer
            context.PerformOnSelf(_preactivation[i]).ApplySigmoidFunction(); // activation of the current layer

            if (computeGradients && IsHiddenLayer(i + 1))
            {
                context.PerformOn(last).And(_weightsTransposed[i]).MultiplyInto(_preactivationGradient[i]);
                context.PerformOnSelf(_preactivationGradient[i]).ApplySigmoidGradientFunction();
            }
            
            if (IsNotOutputLayer(i + 1)) // no bias term added for the output layer
            {
                context.PerformOn(_preactivation[i]).Into(_activation[i]).InsertColumn(1.0); // fully activated layer (added bias)
                last = _activation[i];
            }
            else
            {
                last = _preactivation[i];
            }
        }

        return last; // Output layer predictions
    }
    
    /// <summary>
    /// Vectorized impl of backpropagation 
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    private void Backpropagation(MatrixStorage x, MatrixStorage y)
    {
        ResetErrorTerms();
        
        var context = MatrixComputeContext.Create(MatrixStorage.Operations);
        var prediction = FeedForward(x, true);

        context.PerformOn(prediction).And(y).PointwiseSubtractInto(_errors[1]);
        for (int i = OutputLayerIdx - 1; i >= InputLayerIdx + 1; --i)
        {
            context.PerformOn(_errors[i]).And(_weights[i]).MultiplyInto(_errorsBiased[i - 1]);
            context.PerformOn(_preactivationGradient[i - 1]).Into(_preactivationGradientBiased[i - 1]).InsertColumn(1.0);

            context.PerformOn(_errorsBiased[i - 1]).And(_preactivationGradientBiased[i - 1]).PointwiseMultiplyInto(_errorsBiased[i - 1]);
            context.PerformOn(_errorsBiased[i - 1]).Into(_errors[i - 1]).RemoveColumn();
        }

        var layer = x;
        int samples = x.Rows;
        for (int i = 0; i < _errors.Length; ++i)
        {
            context.PerformOn(_errors[i]).Into(_errorsTransposed[i]).Transpose();
            context.PerformOn(_errorsTransposed[i]).And(layer).MultiplyInto(_derivatives[i]);
            context.PerformOnSelf(_derivatives[i]).MultiplyBy(1.0 / samples);
            
            layer = _preactivation[i];
        }
    }

    public double ComputeCost(MatrixStorage x, MatrixStorage y)
    {
        var hypothesis = FeedForward(x);

        const int hypothesisIdx = 0;
        const int yIdx = 1;
        
        y.Multiply(-1.0, _negativeOutput[yIdx]);
        hypothesis.Multiply(-1.0, _negativeOutput[hypothesisIdx]);

        hypothesis.PointwiseLog(hypothesis);
        _negativeOutput[yIdx].PointwiseMultiply(hypothesis, hypothesis);

        _negativeOutput[yIdx].Add(1.0, _negativeOutput[yIdx]);
        _negativeOutput[hypothesisIdx].Add(1.0, _negativeOutput[hypothesisIdx]);
        _negativeOutput[hypothesisIdx].PointwiseLog(_negativeOutput[hypothesisIdx]);
        _negativeOutput[yIdx].PointwiseMultiply(_negativeOutput[hypothesisIdx], _negativeOutput[yIdx]);
        
        hypothesis.Subtract(_negativeOutput[yIdx], hypothesis);
        
        int m = x.Rows;
        double cost = hypothesis.Sum();
        
        return cost / m;
    }

    public void GradientDescent(MatrixStorage x, MatrixStorage y)
    {
        var alpha = 0.25;
        var iterations = 2000;
    
        for (int i = 0; i < iterations; ++i)
        {
            for (int j = 0; j < _weights.Length; ++j)
            {
                var w = _weights[j];
                _tempWeights[j].Buffer.Reset();
    
                w.Subtract(_derivatives[j], _tempWeights[j], alpha);
            }

            (_weights, _tempWeights) = (_tempWeights, _weights);

            for (int j = 0; j < _weights.Length; ++j)
                _weights[j].Transpose(_weightsTransposed[j]);
            
            if (i % 200 == 0) 
                Console.WriteLine($"Iteration {i}, Cost: {ComputeCost(x, y)}");
            Backpropagation(x, y);
        }
    }
    
    private void ResetErrorTerms()
    {
        int length = _derivatives.Length;
        for (int i = 0; i < length; ++i)
            _derivatives[i].Buffer.Reset();
    }
}