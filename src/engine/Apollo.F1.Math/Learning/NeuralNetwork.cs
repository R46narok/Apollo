using System.Formats.Asn1;
using Apollo.F1.Math.Common;
using Apollo.F1.Math.Common.Interfaces;
using Apollo.F1.Math.Common.LinearAlgebra;
using Apollo.F1.Math.Exceptions;
using Apollo.F1.Math.Neural;

namespace Apollo.F1.Math.Learning;

public class NeuralNetwork : ICostFunction
{
    private readonly int[] _layers;
    private readonly double _learningRate;
    private readonly double _regularizationTerm;
    private readonly double _distributionUpperBound;
    private readonly double _distributionLowerBound;

    public Matrix[] _weights = null!;
    private Matrix[] _tempWeights = null!;
    private Matrix[] _weightsTransposed = null!;
    private Matrix[] _derivatives = null!;
    
    public NeuralNetwork(NeuralNetworkOptions options)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));

        _layers = options.Layers;
        _learningRate = options.LearningRate;
        _regularizationTerm = options.RegularizationTerm;
        _distributionLowerBound = options.DistributionBoundaries.Item1;
        _distributionUpperBound = options.DistributionBoundaries.Item2;
        
        ValidateNetworkArchitecture();
        InitializeWeights();
        InitializeDerivatives();
        InitializeBiasNeurons();
    }

    private void ValidateNetworkArchitecture()
    {
        if (_layers.Length < 2) throw new ArchitectureException();
    }

    private double UniformDistribution(double low, double high) 
    { 
        double difference = high - low; // The difference between the two
        int scale = 100;
       	int scaled_difference = (int)(difference * scale);
       	return low + (1.0 * (Random.Shared.Next() % scaled_difference) / scale);
    }

    private void InitializeWeights()
    {
        int length = _layers.Length - 1;
        _weights = new Matrix[length];
        _weightsTransposed = new Matrix[length];
        _tempWeights = new Matrix[length];
        
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

            _weights[i] = new Matrix(_layers[i + 1], _layers[i] + 1);
            _weights[i].Buffer.Upload(cpuBuffer);
            _tempWeights[i] = new Matrix(_layers[i + 1], _layers[i] + 1);
            _weightsTransposed[i] = _weights[i].Transpose();
        }
    }

    private void InitializeDerivatives()
    {
        int length = _layers.Length - 1;
        _derivatives = new Matrix[length];

        for (int i = 0; i < length; ++i)
            _derivatives[i] = new Matrix(_layers[i + 1], _layers[i] + 1);
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

    private Matrix _z2;
    private Matrix _z2Gradient;
    private Matrix _z2GradientBiased;
    private Matrix _z3;
    private Matrix _a2;
    private Matrix _delta3;
    private Matrix _delta3Transposed;
    private Matrix _delta2Transposed;
    private Matrix _delta2;
    private Matrix _delta2Biased;
    private Matrix _hNegative;
    private Matrix _yNegative;
    
    private void InitFF(Matrix x)
    {
        _z2 = new Matrix(x.Rows, _weightsTransposed[0].Columns);
        _z2Gradient = new Matrix(x.Rows, _weightsTransposed[0].Columns);
        _z2GradientBiased = new Matrix(x.Rows, _weightsTransposed[0].Columns + 1);
        _a2 = new Matrix(x.Rows, _weightsTransposed[0].Columns + 1);
        _z3 = new Matrix(x.Rows, _weightsTransposed[1].Columns);

        _delta3 = new Matrix(x.Rows, _weightsTransposed[1].Columns);
        _delta3Transposed = new Matrix(_delta3.Columns, _delta3.Rows);
        _delta2 = new Matrix(_delta3.Rows, _weights[1].Columns - 1);
        _delta2Transposed = new Matrix(_delta2.Columns, _delta2.Rows);
        _delta2Biased = new Matrix(_delta3.Rows, _weights[1].Columns);

        _hNegative = new Matrix(x.Rows, _weights[1].Rows);
        _yNegative = new Matrix(x.Rows, _weights[1].Rows);
    }

    
    /// <summary>
    /// Computes the output of a neural network based on a given input
    /// using the forward propagation algorithm.
    /// </summary>
    /// <param name="x">Input values(requires already added bias value)</param>
    /// <returns>[training samples x number of output units] matrix, which contains the predictions</returns>
    public Matrix FeedForward(Matrix x)
    {
        if(_z2 is null) InitFF(x);

        var a1 = x;
        
        a1.Multiply(_weightsTransposed[0], _z2);
        _z2.ApplySigmoid(_z2);
        var a2 = _z2;
        
        a2.InsertColumn(1.0, _a2);
        
        _a2.Multiply(_weightsTransposed[1], _z3);
        _z3.ApplySigmoid(_z3);
        return _z3;
    }
    
    // ReSharper disable once IdentifierTypo
    public void Backpropagate(Matrix x, Matrix y)
    {
        if(_z2 is null) InitFF(x);
        int l = _layers.Length;
        int m = x.Rows;

        ResetErrorTerms();

        // Vectorized implementation of backpropagation
        var a1 = x; // x must include the bias column
        a1.Multiply(_weightsTransposed[0], _z2);
        a1.Multiply(_weightsTransposed[0], _z2Gradient);
        _z2.ApplySigmoid(_z2);
        _z2Gradient.ApplySigmoidGradient(_z2Gradient);
        
        var a2 = _z2;
        a2.InsertColumn(1.0, _a2);

        _a2.Multiply(_weightsTransposed[1], _z3);
        _z3.ApplySigmoid(_z3);
        var a3 = _z3;

        a3.Subtract(y, _delta3);
        _delta3.Multiply(_weights[1], _delta2Biased);

        _z2Gradient.InsertColumn(1.0, _z2GradientBiased);
        _delta2Biased.PointwiseMultiply(_z2GradientBiased, _delta2Biased);
        _delta2Biased.RemoveColumn(_delta2);

        _delta2.Transpose(_delta2Transposed);
        _delta2Transposed.Multiply(a1, _derivatives[0]);

        _delta3.Transpose(_delta3Transposed);
        _delta3Transposed.Multiply(a2, _derivatives[1]);

        _derivatives[0].Multiply(1.0 / m);
        _derivatives[1].Multiply(1.0 / m);
    }

    
    public double ComputeCost(Matrix x, Matrix y)
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
        
        h.Subtract(_yNegative);
        cost = h.Sum();
        
        return cost + (0.25/(2 * m));
    }

    public void ComputeDerivatives(Matrix x, Matrix y)
    {
        Backpropagate(x, y);
    }

    public void GradientDescent(Matrix x, Matrix y)
    {
        var alpha = 0.25;
        var iterations = 10000;
    
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