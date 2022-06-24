using Apollo.F1.Math.Exceptions;
using Apollo.F1.Math.Extensions;
using Apollo.F1.Math.Functions;
using MathNet.Numerics.LinearAlgebra;

namespace Apollo.F1.Math.Neural;

public class NeuralNetwork
{
    private readonly int[] _layers;
    private Matrix<double>[] _weights = null!;
    
    public NeuralNetwork(int[] layers)
    {
        if (layers.Length <= 2)
            throw new ArchitectureException();

        _layers = layers;

        InitWeights();
        InitBiasNeurons();
    }

    private void InitWeights()
    {
        int l = _layers.Length - 1;
        
        _weights = new Matrix<double>[l];
        var builder = Matrix<double>.Build;
        for (int i = 0; i < l; ++i)
        {
            _weights[i] = i != l - 1 ? 
                builder.Random(_layers[i + 1], _layers[i] + 1) 
                : builder.Random(_layers[i + 1], _layers[i]);
        }
    }

    private void InitBiasNeurons()
    {
        int l = _layers.Length - 1;
        for (int i = 0; i < l; ++i)
        {
            _layers[i]++;
            
            var matrix = _weights[i];
            for (int j = 0; j < matrix.RowCount; ++j)
                matrix[j, 0] = 1.0;
        }
    }
    
    public Matrix<double> ForwardPropagation(Vector<double> x, Vector<double>[]? cache)
    {
        int l = _layers.Length - 1;
    
        Matrix<double> activation = x.ToColumnMatrix();
        for (int i = 0; i < l; ++i)
        {
            activation = _weights[i].Multiply(activation);
            activation.Apply(Activation.Sigmoid);

            if (cache is not null && i < cache.Length)
            {
                cache[i] = activation.Column(0);
            }
        }
        
        return activation;
    }
    
    public void Backpropagation(Matrix<double> x, Matrix<double> y)
    {
        int l = _layers.Length;
        int m = x.RowCount;
        
        var M = Matrix<double>.Build;
        var V = Vector<double>.Build;

        var delta = new Matrix<double>[l];
        for (int i = 1; i < l; ++i)
        {
            delta[i - 1] = M.Dense(m, _layers[i]);
        }
    
        // For every training sample
        for (int i = 0; i < m; ++i)
        {
            // Init errors
            var error = new Vector<double>[l - 1];
            for (int j = 1; j < l; ++j)
                error[j - 1] = V.Dense(m);
            
            // Compute activation of every neuron in the neural network
            // and the output a(L)
            var a = new Vector<double>[l - 1];
            ForwardPropagation(x.Row(i), a);
            
            // Compute the the last set of errors 
            var expectedOutput = y.Row(i);
            error[l - 2] = a[l - 2].Subtract(expectedOutput);
            
            // Compute the rest of the errors
            for (int j = error.Length - 2; j >= 0; --j)
            {
                var idk = _weights[j + 1].Multiply(error[j + 1]);
            }
        }
    }
    
    public double ComputeCost(Matrix<double> x, Matrix<double> y)
    {
        double cost = 0.0;
        int m = x.RowCount;
    
        for (int i = 0; i < m; ++i)
        {
            var sample = x.Row(i);
            var hypothesis = ForwardPropagation(sample, null);
    
            for (int k = 0; k < hypothesis.RowCount; ++k)
            {
                var output = hypothesis[k, 0];
                var expectedOutput = y[i, k];
    
                cost += expectedOutput * System.Math.Log10(output) +
                        (1 - expectedOutput) * System.Math.Log10(1 - output);
            }
        }
    
        cost *= -1 * (1 / m);
    
        double regularization = 0.0;
        int L = _layers.Length - 1;
        for (int l = 1; l < L; ++l)
        {
            for (int i = 0; i < _weights[l].RowCount; ++i)
            {
                for (int j = 1; j < _weights[l].ColumnCount; ++j)
                {
                    regularization += System.Math.Pow(_weights[l][i, j], 2);
                }
            }
        }
    
        regularization *= 0.25 * (1 / (2 * m));
        
        return cost + regularization;
    }
}