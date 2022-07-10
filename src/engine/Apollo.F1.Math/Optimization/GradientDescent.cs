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
        
    }
}