namespace Apollo.F1.Compute.Neural;

public class NeuralNetworkOptions
{
    public const double DefaultLearningRate = 0.25;
    public const double DefaultRegularizationTerm = 0.25;
    public static (double, double) DefaultDistributionBoundaries => (-1.0 * System.Math.Sqrt(6), System.Math.Sqrt(6)); 
    
    public int[] Layers { get; set; }
    public double LearningRate { get; set; } = DefaultLearningRate; // Alpha 
    public double RegularizationTerm { get; set; } = DefaultRegularizationTerm; // Lambda
    public (double, double) DistributionBoundaries { get; set; } = DefaultDistributionBoundaries;
}