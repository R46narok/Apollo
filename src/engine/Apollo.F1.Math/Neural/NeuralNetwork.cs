using Apollo.F1.Math.Exceptions;
using Apollo.F1.Math.Functions;

namespace Apollo.F1.Math.Neural;

public class NeuralNetwork
{
    private readonly uint[] _layers;
    private Matrix[] _weights = null!;
    
    public NeuralNetwork(uint[] layers)
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
        
        _weights = new Matrix[l];
        for (int i = 0; i < l; ++i)
        {
            _weights[i] = i != l - 1 ? 
                new Matrix(_layers[i + 1], _layers[i] + 1) 
                : new Matrix(_layers[i + 1], _layers[i]);
            _weights[i].Randomize(2);
        }
    }

    private void InitBiasNeurons()
    {
        int l = _layers.Length - 1;
        for (int i = 0; i < l; ++i)
        {
            _layers[i]++;
            
            var matrix = _weights[i];
            for (int j = 0; j < matrix.Rows; ++j)
                matrix[j, 0] = 1.0;
        }
    }

    public Matrix ForwardPropagation(Matrix x)
    {
        int l = _layers.Length - 1;

        Matrix activation = x;
        for (int i = 0; i < l; ++i)
        {
            activation = _weights[i].Dot(activation);
            activation.Apply(Activation.Sigmoid);
        }
        
        return activation;
    }

    public void Backpropagation(Matrix x, Matrix y)
    {
        int l = _layers.Length;
        uint trainingLength = y.Rows;
        
        var delta = new Matrix[l];
        for (int i = 1; i < l; ++i)
        {
            delta[i] = new Matrix(trainingLength, _layers[i]);
        }
    }
    
    public double ComputeCost(Matrix x, Matrix y)
    {
        double cost = 0.0;
        uint m = x.Rows;

        for (uint i = 0; i < m; ++i)
        {
            var rowSpan = x.AsRowVector(i);
            var rowAsMatrix = new Matrix(rowSpan.ToArray(), 1, (uint) rowSpan.Length).Flatten(VectorType.Column);
            var hypothesis = ForwardPropagation(rowAsMatrix);

            for (uint k = 0; k < hypothesis.Rows; ++k)
            {
                var output = hypothesis[0, k];
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
            for (int i = 0; i < _weights[l].Rows; ++i)
            {
                for (int j = 1; j < _weights[l].Columns; ++j)
                {
                    regularization += System.Math.Pow(_weights[l][i, j], 2);
                }
            }
        }

        regularization *= 0.25 * (1 / (2 * m));
        
        return cost + regularization;
    }
}