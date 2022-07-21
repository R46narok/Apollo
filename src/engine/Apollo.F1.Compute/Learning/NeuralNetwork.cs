using System.ComponentModel.DataAnnotations;
using System.Formats.Asn1;
using Apollo.F1.Compute.Common.Interfaces;
using Apollo.F1.Compute.Common.LinearAlgebra;
using Apollo.F1.Compute.Exceptions;
using Apollo.F1.Compute.Neural;
using Apollo.F1.Compute.Common;
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
    
    public MatrixStorage[] _weights = null!;
    private MatrixStorage[] _tempWeights = null!;
    public MatrixStorage[] _weightsTransposed = null!;
    public MatrixStorage[] _derivatives = null!;
    
    public NeuralNetwork(NeuralNetworkOptions options)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));

        _layers = options.Layers;
        _learningRate = options.LearningRate;
        _regularizationTerm = options.RegularizationTerm;
        _distributionLowerBound = options.DistributionBoundaries.Item1;
        _distributionUpperBound = options.DistributionBoundaries.Item2;
        _matrixOperations = MatrixStorage.Operations;
        
        ValidateNetworkArchitecture();
        InitializeWeights();
        InitializeDerivatives();
        InitializeBufferBatches(700);
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

    public MatrixStorage _z2;
    public MatrixStorage _z2Gradient;
    public MatrixStorage _z2GradientBiased;
    public MatrixStorage _z3;
    public MatrixStorage _a2;
    public MatrixStorage _delta3;
    public MatrixStorage _delta3Transposed;
    public MatrixStorage _delta2Transposed;
    public MatrixStorage _delta2;
    public MatrixStorage _delta2Biased;
    public MatrixStorage _hNegative;
    public MatrixStorage _yNegative;
    
    public void InitFF(MatrixStorage x)
    {
        _z2 = new MatrixStorage(x.Rows, _weightsTransposed[0].Columns);
        _z2Gradient = new MatrixStorage(x.Rows, _weightsTransposed[0].Columns);
        _z2GradientBiased = new MatrixStorage(x.Rows, _weightsTransposed[0].Columns + 1);
        _a2 = new MatrixStorage(x.Rows, _weightsTransposed[0].Columns + 1);
        _z3 = new MatrixStorage(x.Rows, _weightsTransposed[1].Columns);

        _delta3 = new MatrixStorage(x.Rows, _weightsTransposed[1].Columns);
        _delta3Transposed = new MatrixStorage(_delta3.Columns, _delta3.Rows);
        _delta2 = new MatrixStorage(_delta3.Rows, _weights[1].Columns - 1);
        _delta2Transposed = new MatrixStorage(_delta2.Columns, _delta2.Rows);
        _delta2Biased = new MatrixStorage(_delta3.Rows, _weights[1].Columns);

        _hNegative = new MatrixStorage(x.Rows, _weights[1].Rows);
        _yNegative = new MatrixStorage(x.Rows, _weights[1].Rows);
    }

    private BufferBatch _preactivationBatch;
    private BufferBatch _activationBatch;
    private BufferBatch _errorBatch;
    private BufferBatch _errorsTransposedBatch;
    private MatrixStorage[] _preactivation = null!;
    private MatrixStorage[] _activation;
    private MatrixStorage[] _errors = null!;
    private MatrixStorage[] _errorsTransposed = null!;

    private void InitializeBufferBatches(int samples)
    {
        InitializeBatchAsMatrixArray(out _preactivationBatch, out _preactivation, _layers.Length - 1,
            i => samples, i => _weightsTransposed[i].Columns, BufferDataType.Double, "preactivation");
        
        InitializeBatchAsMatrixArray(out _activationBatch, out _activation, _layers.Length - 1,
            i => samples, i=> _weightsTransposed[i].Columns + 1, BufferDataType.Double, "activation");
        
        InitializeBatchAsMatrixArray(out _errorBatch, out _errors, _layers.Length - 1,
            i => samples, i => _weightsTransposed[i].Columns, BufferDataType.Double, "errors");
        
        InitializeBatchAsMatrixArray(out _errorsTransposedBatch, out _errorsTransposed, _layers.Length - 1,
            i => _weightsTransposed[i].Columns, i => samples, BufferDataType.Double, "errorsTransposed");
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

    /// <summary>
    /// Computes the output of a neural network based on a given input
    /// using the forward propagation algorithm.
    /// </summary>
    /// <param name="x">Input values(requires already added bias value)</param>
    /// <returns>[training samples x number of output units] matrix, which contains the predictions</returns>
    public MatrixStorage FeedForward(MatrixStorage x)
    {
        if(_preactivation is null) InitializeBufferBatches(700);

        var context = MatrixComputeContext.Create(MatrixStorage.Operations);
        var a1 = x;

        context = context
            .PerformOn(a1)
            .And(_weightsTransposed[0])
            .MultiplyInto(_preactivation[0]);
        
        context = context
            .PerformOnSelf(_preactivation[0])
            .ApplySigmoidFunction();

        context = context
            .PerformOn(_preactivation[0])
            .Into(_activation[0])
            .InsertColumn(1.0);

        context = context
            .PerformOn(_activation[0])
            .And(_weightsTransposed[1])
            .MultiplyInto(_preactivation[1]);

        context = context
            .PerformOnSelf(_preactivation[1])
            .ApplySigmoidFunction();
        
        return _preactivation[1];
    }
    
    // ReSharper disable once IdentifierTypo
    public void Backpropagate(MatrixStorage x, MatrixStorage y)
    {
        if(_z2 is null) InitFF(x);

        ResetErrorTerms();

        var context = MatrixComputeContext.Create(MatrixStorage.Operations);
        
        // Vectorized implementation of backpropagation
        var a1 = x;
        context = context
            .PerformOn(a1)
            .And(_weightsTransposed[0])
            .MultiplyInto(_preactivation[0]);

        context = context
            .PerformOn(a1)
            .And(_weightsTransposed[0])
            .MultiplyInto(_z2Gradient);

        context = context
            .PerformOnSelf(_preactivation[0])
            .ApplySigmoidFunction();

        context = context
            .PerformOn(_preactivation[0])
            .Into(_z2Gradient)
            .ApplySigmoidGradientFunction();

        context = context
            .PerformOn(_preactivation[0])
            .Into(_activation[0])
            .InsertColumn(1.0);

        context = context
            .PerformOn(_activation[0])
            .And(_weightsTransposed[1])
            .MultiplyInto(_preactivation[1]);
        
        context = context
            .PerformOnSelf(_preactivation[1])
            .ApplySigmoidFunction();
        
        var a3 = _preactivation[1];

        context = context
            .PerformOn(a3)
            .And(y)
            .PointwiseSubtractInto(_errors[1]);

        context = context
            .PerformOn(_errors[1])
            .And(_weights[1])
            .MultiplyInto(_delta2Biased);

        context = context
            .PerformOn(_z2Gradient)
            .Into(_z2GradientBiased)
            .InsertColumn(1.0);

        _delta2Biased.PointwiseMultiply(_z2GradientBiased, _delta2Biased);
        _delta2Biased.RemoveColumn(_delta2);
        
        /*context = context
            .PerformOn(_delta2Biased)
            .And(_z2GradientBiased)
            .PointwiseMultiplyInto(_delta2Biased);*/

        /*context = context
            .PerformOn(_delta2Biased)
            .Into(_delta2)
            .RemoveColumn();*/

         //_delta2.Transpose(_delta2Transposed);
         //_delta2Transposed.Multiply(a1, _derivatives[0]);
        
        context = context
            .PerformOn(_delta2)
            .Into(_delta2Transposed)
            .Transpose();

        context = context
            .PerformOn(_delta2Transposed)
            .And(a1)
            .MultiplyInto(_derivatives[0]);
        
        context = context
            .PerformOn(_delta3)
            .Into(_delta3Transposed)
            .Transpose();

        context = context
            .PerformOn(_delta3Transposed)
            .And(_preactivation[0])
            .MultiplyInto(_derivatives[1]);
        
        int m = x.Rows;
        _derivatives[0].Multiply(1.0 / m);
        _derivatives[1].Multiply(1.0 / m);
    }

    public double ComputeCost(MatrixStorage x, MatrixStorage y)
    {
        if(_z2 is null) InitFF(x);
        double cost = 0.0;
        int m = x.Rows;

        var h = FeedForward(x);
        
        y.Multiply(-1.0, _yNegative);
        h.Multiply(-1.0, _hNegative);

        h.PointwiseLog(h);
        _yNegative.PointwiseMultiply(h, h);

        _yNegative.Add(1.0, _yNegative);
        _hNegative.Add(1.0, _hNegative);
        _hNegative.PointwiseLog(_hNegative);
        _yNegative.PointwiseMultiply(_hNegative, _yNegative);
        
        h.Subtract(_yNegative, h);
        cost = h.Sum();
        
        return cost / (double)m;
    }

    public MatrixStorage[] ComputeDerivatives(MatrixStorage x, MatrixStorage y)
    {
        Backpropagate(x, y);
        return _derivatives;
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
            Backpropagate(x, y);
        }
    }
    
    private void ResetErrorTerms()
    {
        int length = _derivatives.Length;
        for (int i = 0; i < length; ++i)
            _derivatives[i].Buffer.Reset();
    }
}