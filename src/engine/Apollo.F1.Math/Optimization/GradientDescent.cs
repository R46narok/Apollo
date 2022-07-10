using Apollo.F1.Math.Common.Interfaces;
using Apollo.F1.Math.Common.LinearAlgebra;

namespace Apollo.F1.Math.Optimization;

public class GradientDescent : IOptimizationProcedure
{
    private readonly double _learningRate;
    private readonly int _iterations;

    public GradientDescent(double learningRate, int iterations)
    {
        _learningRate = learningRate;
        _iterations = iterations;
    }

    public void Optimize(ICostFunction function, Matrix[] parameters, Matrix[] derivatives, Matrix x, Matrix y)
    {
        for (int i = 0; i < _iterations; ++i)
        {
            var temp = new Matrix[parameters.Length];
        
            for (int j = 0; j < parameters.Length; ++j)
            {
                var w = parameters[j];
                temp[j] = new Matrix(w.Rows, w.Columns);
        
                w.Subtract(derivatives[j], temp[j], _learningRate);
            }
            parameters = temp;
            Console.WriteLine($"Iteration {i}, Cost: {function.ComputeCost(x, y)}");
            function.ComputeDerivatives(x, y);
        }
    }
}