using Apollo.F1.Compute.Common.LinearAlgebra;
using Apollo.F1.Compute.Cuda.Buffers;
using Apollo.F1.Compute.Cuda.Common;

namespace Apollo.F1.Compute.Cuda.Kernels;

public class PointwiseKernelOptions : KernelOptionsBase
{
    public GpuBuffer FirstOperand { get; set; }
    public GpuBuffer SecondOperand { get; set; }
    public double Scale { get; set; }

    public PointwiseKernelOptions(Matrix first, Matrix second, Matrix output, double scale = 1.0)
    {
        FirstOperand = first.Buffer as GpuBuffer;
        SecondOperand = second.Buffer as GpuBuffer;
        Output = output.Buffer as GpuBuffer;
        Scale = scale;
    }
}

public class PointwiseOperationKernelOptions : KernelOptionsBase
{
    public GpuBuffer Operand { get; set; }

    public PointwiseOperationKernelOptions(Matrix input, Matrix output)
    {
        Operand = input.Buffer as GpuBuffer;
        Output = output.Buffer as GpuBuffer;
    }
}