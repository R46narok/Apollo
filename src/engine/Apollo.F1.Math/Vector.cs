using Apollo.F1.Math.Exceptions;

namespace Apollo.F1.Math;

public enum VectorType
{
    Row,
    Column
}

public class Vector
{
    public static double Transpose(ReadOnlySpan<double> left, ReadOnlySpan<double> right)
    {
        if (left.Length != right.Length)
            throw new DimensionsException();

        double sum = 0.0;
        for (int i = 0; i < left.Length; ++i)
            sum += left[i] * right[i];

        return sum;
    }
}