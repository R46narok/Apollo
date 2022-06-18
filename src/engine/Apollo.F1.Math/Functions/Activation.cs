namespace Apollo.F1.Math.Functions;

public static class Activation
{
    public static double Sigmoid(double input)
    {
        return 1.0 / (1 + System.Math.Exp(-1 * input));
    }
}