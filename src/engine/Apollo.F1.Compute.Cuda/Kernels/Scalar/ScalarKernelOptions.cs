using Apollo.F1.Compute.Common.LinearAlgebra;
using Apollo.F1.Compute.Cuda.Buffers;
using Apollo.F1.Compute.Cuda.Common;

namespace Apollo.F1.Compute.Cuda.Kernels;

public class ScalarKernelOptions : KernelOptionsBase
{
    public GpuBuffer Input { get; set; }
    public double Scalar { get; set; }

    public ScalarKernelOptions(Matrix input, Matrix output, double scalar)
    {
        Input = input.Buffer as GpuBuffer;
        Output = output.Buffer as GpuBuffer;
        Scalar = scalar;
    }
}