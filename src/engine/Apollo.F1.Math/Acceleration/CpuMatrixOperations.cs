namespace Apollo.F1.Math.Acceleration;

public static class CpuMatrixOperations
{
    private static Random _random = new Random();

    private static double UniformDistribution(double low, double high)
    {
        double diff = high - low;
        const int scale = 10000;
        int scaledDifference = (int) (diff * scale);

        return low + (1.0 * (_random.Next() % scaledDifference) / scale);
    }
    
    public static void Randomize(Span<double> dst, int n)
    {
        double min = -1.0 / System.Math.Sqrt(n);
        double max = 1.0 / System.Math.Sqrt(n);

        for (int i = 0; i < dst.Length; ++i)
        {
            dst[i] = UniformDistribution(min, max);
        }
    }
    
    public static void Add(Span<double> dst, ReadOnlySpan<double> first, ReadOnlySpan<double> second)
    {
        for (int i = 0; i < first.Length; ++i)
            dst[i] = first[i] + second[i];
    }
    
    public static void Subtract(Span<double> dst, ReadOnlySpan<double> first, ReadOnlySpan<double> second)
    {
        for (int i = 0; i < first.Length; ++i)
            dst[i] = first[i] - second[i];
    }
}