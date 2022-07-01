using System.Reflection;
using Apollo.F1.Math.Exceptions;
using Apollo.F1.Math.Extensions;
using Apollo.F1.Math.Functions;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace Apollo.F1.Math.Neural;

public class NeuralNetwork
{
    private readonly int[] _layers;
    private Matrix<double>[] _weights = null!;
    private Matrix<double>[] _delta = null!;
    
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
        _delta = new Matrix<double>[l];
        var builder = Matrix<double>.Build;
        for (int i = 0; i < l; ++i)
        {
            var bound = System.Math.Sqrt(6);
            _weights[i] = builder.Random(_layers[i + 1], _layers[i] + 1, new ContinuousUniform(-1.0 * bound, bound));
            _delta[i] = builder.Dense(_layers[i + 1], _layers[i] + 1, 0);
        }
    }

    private void InitBiasNeurons()
    {
        int l = _layers.Length - 1;
        for (int i = 0; i < l; ++i)
        {
            _layers[i]++;
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
            
            if (cache is not null)
            {
                cache[i] = activation.Column(0);
            }
            if (i != l - 1)
                activation = activation.InsertRow(0,
                Vector<double>.Build.Dense(1, 1.0));
        }
        
        return activation;
    }
    
    public void Backpropagation(Matrix<double> x, Matrix<double> y)
    {
        x = x.InsertColumn(0, 
            Vector<double>.Build.Dense(x.RowCount, 1.0)); // Add bias value
        
        int l = _layers.Length;
        int m = x.RowCount;
        
        var M = Matrix<double>.Build;
        var V = Vector<double>.Build;

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
            var delta2 = delta3.Multiply(_weights[1]).ElementwiseMultiplication(a2.Transpose());

            delta2 = delta2.RemoveColumn(0);
            
            _delta[0] = _delta[0] + (a1.Multiply(delta2)).Transpose();
            _delta[1] = _delta[1] + (a2.Multiply(delta3)).Transpose();
        }

        _delta[0] = _delta[0] * (1.0 / m);
        _delta[1] = _delta[1] * (1.0 / m);
    }

    public void GradientDescent(Matrix<double> x, Matrix<double> y)
    {
        var alpha = 0.25;
        var iterations = 100;

        for (int i = 0; i < iterations; ++i)
        {
            var temp = new Matrix<double>[_weights.Length];

            for (int j = 0; j < _weights.Length; ++j)
            {
                var w = _weights[j];
                temp[j] = Matrix<double>.Build.Dense(w.RowCount, w.ColumnCount);
                
                for (int k = 0; k < w.RowCount; ++k)
                for (int l = 0; l < w.ColumnCount; ++l)
                    temp[j][k, l] = w[k, l] - alpha * _delta[j][k, l];
            }
            _weights = temp;
            Console.WriteLine($"Iteration {i}, Cost: {ComputeCost(x, y)}");
            Backpropagation(x, y);
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
        
        return cost + (0.25/(2 * m));
    }
}