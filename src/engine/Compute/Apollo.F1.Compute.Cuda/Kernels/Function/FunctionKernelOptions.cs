using Apollo.F1.Compute.Common.LinearAlgebra;
using Apollo.F1.Compute.Cuda.Buffers;
using Apollo.F1.Compute.Cuda.Common;
using Apollo.F1.Compute.Cuda.Common.Execution;

namespace Apollo.F1.Compute.Cuda.Kernels;

public class FunctionKernelOptions : KernelOptionsBase
{
    public GpuBuffer Input { get; set; }

    public FunctionKernelOptions(MatrixStorage input, MatrixStorage output)
    {
        Input = input.Buffer as GpuBuffer;
        Output = output.Buffer as GpuBuffer;
    }
}