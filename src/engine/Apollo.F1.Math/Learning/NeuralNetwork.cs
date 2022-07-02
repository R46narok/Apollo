using Apollo.F1.Math.Exceptions;
using Apollo.F1.Math.Extensions;
using Apollo.F1.Math.Functions;
using Apollo.F1.Math.Neural;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace Apollo.F1.Math.Learning;

public class NeuralNetwork
{
    private readonly int[] _layers;
    private readonly double _learningRate;
    private readonly double _regularizationTerm;
    private readonly double _distributionUpperBound;
    private readonly double _distributionLowerBound;
    
    private Matrix<double>[] _weights = null!;
    private Matrix<double>[] _derivatives = null!;
    
    private readonly MatrixBuilder<double> M = Matrix<double>.Build;
    private readonly VectorBuilder<double> V = Vector<double>.Build;

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

    private void InitializeWeights()
    {
        int length = _layers.Length - 1;
        _weights = new Matrix<double>[length];
        
        for (int i = 0; i < length; ++i)
        {
            var uniform = new ContinuousUniform(_distributionLowerBound, _distributionUpperBound);
            _weights[i] = M.Random(_layers[i + 1], _layers[i] + 1, uniform);
        }
    }

    private void InitializeDerivatives()
    {
        int length = _layers.Length - 1;
        _derivatives = new Matrix<double>[length];

        for (int i = 0; i < length; ++i)
            _derivatives[i] = M.Dense(_layers[i + 1], _layers[i] + 1, M.Zero);
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
    /// <param name="preactivation">Storage to cache preactivation values</param>
    /// <param name="activation">Storage to cache activation values</param>
    /// <returns>[1 x number of output units] matrix, which contains the predictions</returns>
    public Matrix<double> FeedForward(Vector<double> x, Vector<double>[]? preactivation = null, Vector<double>[]? activation = null)
    {
        int length = _layers.Length - 1;
    
        Matrix<double> a = x.ToColumnMatrix();
        for (int i = 0; i < length; ++i)
        {
            a = _weights[i].Multiply(a);
            if (preactivation is not null) preactivation[i] = a.Column(0);
            
            // Activate using the sigmoid function
            a = a.Map(Activation.Sigmoid);
            if (activation is not null) activation[i] = a.Column(0);
            
            // Not inserting a bias value for the last layer (prediction)
            if (i != length - 1) a = a.InsertBiasRow();
        }
        
        return a;
    }
    
    // ReSharper disable once IdentifierTypo
    public void Backpropagate(Matrix<double> x, Matrix<double> y)
    {
        x = x.InsertBiasColumn();
        
        int l = _layers.Length;
        int m = x.RowCount;
        
        ResetErrorTerms();
        
        // For every training sample
        for (int i = 0; i < m; ++i)
        {
            var a1 = x.Row(i).ToColumnMatrix();
            
            var z2 = _weights[0].Multiply(a1);
            var a2 = z2.Map(Activation.Sigmoid);
            a2 = a2.InsertRow(0,
                V.Dense(1, 1.0));
            
            var z3 = _weights[1].Multiply(a2);
            var a3 = z3.Map(Activation.Sigmoid);

            var yVector = y.Row(i).ToColumnMatrix();
            var delta3 = (a3 - yVector).Transpose();
            //var delta2 = delta3.Multiply(_weights[1]).PointwiseMultiply(a2.Transpose());
            var delta2 = delta3.Multiply(_weights[1]).PointwiseMultiply(z2.Map(Activation.SigmoidGradient).InsertBiasRow().Transpose());
            
            delta2 = delta2.RemoveBiasColumn();
            
            _derivatives[0] = _derivatives[0] + (a1.Multiply(delta2)).Transpose();
            _derivatives[1] = _derivatives[1] + (a2.Multiply(delta3)).Transpose();
        }

        _derivatives[0] = _derivatives[0] * (1.0 / m);
        _derivatives[1] = _derivatives[1] * (1.0 / m);
    }

    public void GradientDescent(Matrix<double> x, Matrix<double> y)
    {
        var alpha = 0.25;
        var iterations = 2000;

        for (int i = 0; i < iterations; ++i)
        {
            var temp = new Matrix<double>[_weights.Length];

            for (int j = 0; j < _weights.Length; ++j)
            {
                var w = _weights[j];
                temp[j] = Matrix<double>.Build.Dense(w.RowCount, w.ColumnCount);
                
                for (int k = 0; k < w.RowCount; ++k)
                for (int l = 0; l < w.ColumnCount; ++l)
                    temp[j][k, l] = w[k, l] - alpha * _derivatives[j][k, l];
            }
            _weights = temp;
            Console.WriteLine($"Iteration {i}, Cost: {ComputeCost(x, y)}");
            Backpropagate(x, y);
        }
    }
    
    public double ComputeCost(Matrix<double> x, Matrix<double> y)
    {
        double cost = 0.0;
        int m = x.RowCount;
    
        var X = x.InsertColumn(0, 
            Vector<double>.Build.Dense(x.RowCount, 1.0));

        var a1 = X;
        
        var z2 = a1.Multiply(_weights[0].Transpose());
        var a2 = z2.Map(f => Activation.Sigmoid(f));

        a2 = a2.InsertColumn(0,
            Vector<double>.Build.Dense(a2.RowCount, 1.0));
        var z3 = a2.Multiply(_weights[1].Transpose());
        var a3 = z3.Map(Activation.Sigmoid);
        var h_x = a3;

        var temp = y.Multiply(-1.0).PointwiseMultiply(h_x.PointwiseLog());
        temp = temp - ((y.Multiply(-1.0) + 1).PointwiseMultiply((h_x.Multiply(-1) + 1).PointwiseLog()));
        cost = temp.ColumnSums().Sum();
        cost /= (double)m;
        
        Backpropagate(x, y);
        
        return cost + (0.25/(2 * m));
    }

    private void ResetErrorTerms()
    {
        int length = _derivatives.Length;
        for (int i = 0; i < length; ++i)
            _derivatives[i] = _derivatives[i].Map(f => 0.0);
    }
}