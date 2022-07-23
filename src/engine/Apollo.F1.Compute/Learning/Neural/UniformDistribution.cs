namespace Apollo.F1.Compute.Learning.Neural;

public class UniformDistribution
{
    private readonly double _bound;

    public UniformDistribution(double bound)
    {
        _bound = bound;
    }

    public double Next()
    {
        var axis = (double)(Random.Shared.Next(0, 2) * 2 - 1);
        var distribution = Random.Shared.NextSingle();
        var value = axis * _bound * distribution;

        return value;
    }
}