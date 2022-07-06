using Apollo.F1.Math.Common.LinearAlgebra;
using Apollo.F1.Math.Cuda.Buffers;
using Apollo.F1.Math.Cuda.Common;

namespace Apollo.F1.Math.Cuda.Kernels;

public class FunctionKernelOptions : KernelOptionsBase
{
    public GpuBuffer Input { get; set; }

    public FunctionKernelOptions(Matrix input, Matrix output)
    {
        Input = input.Buffer as GpuBuffer;
        Output = output.Buffer as GpuBuffer;
    }
}