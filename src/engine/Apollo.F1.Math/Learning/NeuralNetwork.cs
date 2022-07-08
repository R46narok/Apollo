using Apollo.F1.Math.Common.LinearAlgebra;
using Apollo.F1.Math.Exceptions;
using Apollo.F1.Math.Neural;

namespace Apollo.F1.Math.Learning;

public class NeuralNetwork
{
    private readonly int[] _layers;
    private readonly double _learningRate;
    private readonly double _regularizationTerm;
    private readonly double _distributionUpperBound;
    private readonly double _distributionLowerBound;

    private Matrix[] _weights = null!;
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
    
    /// <summary>
    /// Computes the output of a neural network based on a given input
    /// using the forward propagation algorithm.
    /// </summary>
    /// <param name="x">Input values(requires already added bias value)</param>
    /// <returns>[training samples x number of output units] matrix, which contains the predictions</returns>
    public Matrix FeedForward(Matrix x)
    {
        var a1 = x;
                
        var z2 = a1.Multiply(_weights[0].Transpose());
        z2.ApplySigmoid(z2);
        var a2 = z2;

        a2 = a2.InsertColumn(1.0);
        
        var z3 = a2.Multiply(_weights[1].Transpose());
        z3.ApplySigmoid(z3);
        return z3;
    }
    
    // ReSharper disable once IdentifierTypo
    public void Backpropagate(Matrix x, Matrix y)
    {
        int l = _layers.Length;
        int m = x.Rows;

        ResetErrorTerms();

        // Vectorized implementation of backpropagation
        var a1 = x; // x must include the bias column
        var z2 = a1.Multiply(_weights[0].Transpose());
        var z2Gradient = a1.Multiply(_weights[0].Transpose());
        z2.ApplySigmoid(z2);
        z2Gradient.ApplySigmoidGradient(z2Gradient);
        
        var a2 = z2;
        a2 = a2.InsertColumn(1.0);

        var z3 = a2.Multiply(_weights[1].Transpose());
        z3.ApplySigmoid(z3);
        var a3 = z3;

        var delta3 = a3.Subtract(y);
        var delta2 = delta3.Multiply(_weights[1]);

        delta2 = delta2.PointwiseMultiply(z2Gradient.InsertColumn(1.0));
        delta2 = delta2.RemoveColumn();

        _derivatives[0] = delta2.Transpose().Multiply(a1);
        _derivatives[1] = delta3.Transpose().Multiply(a2);

        _derivatives[0].Multiply(1.0 / m);
        _derivatives[1].Multiply(1.0 / m);
    }

    public void GradientDescent(Matrix x, Matrix y)
    {
        var alpha = 0.25;
        var iterations = 2000;

        for (int i = 0; i < iterations; ++i)
        {
            var temp = new Matrix[_weights.Length];

            for (int j = 0; j < _weights.Length; ++j)
            {
                var w = _weights[j];
                temp[j] = new Matrix(w.Rows, w.Columns);

                w.Subtract(_derivatives[j], temp[j], alpha);
            }
            _weights = temp;
            Console.WriteLine($"Iteration {i}, Cost: {ComputeCost(x, y)}");
            Backpropagate(x, y);
        }
    }
    
    public double ComputeCost(Matrix x, Matrix y)
    {
        double cost = 0.0;
        int m = x.Rows;

        var h = FeedForward(x);

        var yNegative = new Matrix(y.Rows, y.Columns);
        var hNegative = new Matrix(h.Rows, h.Columns); 
        y.Multiply(-1.0, yNegative);
        h.Multiply(-1.0, hNegative);
        
        var temp = yNegative.PointwiseMultiply(h.PointwiseLog());
        temp = temp.Subtract(yNegative.Add(1.0).PointwiseMultiply(hNegative.Add(1).PointwiseLog()));
        cost = temp.Sum();
        
        return cost + (0.25/(2 * m));
    }

    private void ResetErrorTerms()
    {
        int length = _derivatives.Length;
        for (int i = 0; i < length; ++i)
            _derivatives[i].Buffer.Reset();
    }
}