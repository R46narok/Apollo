using Apollo.F1.Compute.Common.Interfaces;
using Apollo.F1.Compute.Common.LinearAlgebra;

namespace Apollo.F1.Compute.Optimization;

public class GradientDescent : IOptimizationProcedure
{
    private readonly double _learningRate;
    private readonly int _iterations;
    private MatrixStorage[] _tempParameters = null!;
    private MatrixStorage[] _parametersTransposed = null!;
    
    public GradientDescent(double learningRate, int iterations)
    {
        _learningRate = learningRate;
        _iterations = iterations;
    }

    public void Optimize(ICostFunction function, MatrixStorage[] parameters, MatrixStorage x, MatrixStorage y)
    {
        int length = parameters.Length;
        _tempParameters = new MatrixStorage[length];
        _parametersTransposed = new MatrixStorage[length];
        
        for (int i = 0; i < length; ++i)
        {
            _tempParameters[i] = new MatrixStorage(parameters[i].Rows, parameters[i].Columns);
            _parametersTransposed[i] = new MatrixStorage(parameters[i].Columns, parameters[i].Rows);
        }
        
        for (int i = 0; i < _iterations; ++i)
        {
            var derivatives = function.ComputeDerivatives(x, y);
            for (int j = 0; j < parameters.Length; ++j)
            {
                var w = parameters[j];
                _tempParameters[j].Buffer.Reset();
       
                w.Subtract(derivatives[j], _tempParameters[j], _learningRate);
            }
        
            (parameters, _tempParameters) = (_tempParameters, parameters);
        
            for (int j = 0; j < parameters.Length; ++j)
                parameters[j].Transpose(_parametersTransposed[j]);
            
            Console.WriteLine($"Iteration {i}, Cost: {function.ComputeCost(x, y)}");
        }
    }
}