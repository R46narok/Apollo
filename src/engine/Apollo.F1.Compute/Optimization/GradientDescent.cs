using Apollo.F1.Compute.Common.Interfaces;
using Apollo.F1.Compute.Common.LinearAlgebra;

namespace Apollo.F1.Compute.Optimization;

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