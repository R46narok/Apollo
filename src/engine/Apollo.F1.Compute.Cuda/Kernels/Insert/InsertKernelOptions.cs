using Apollo.F1.Compute.Common.LinearAlgebra;
using Apollo.F1.Compute.Cuda.Buffers;
using Apollo.F1.Compute.Cuda.Common;

namespace Apollo.F1.Compute.Cuda.Kernels;

public class InsertKernelOptions : KernelOptionsBase
{
    public GpuBuffer Input { get; set; }
    public int Rows { get; set; }
    public int Columns { get; set; }
    public double Value { get; set; }

    public InsertKernelOptions(MatrixStorage input, MatrixStorage output, double value)
    {
        Input = input.Buffer as GpuBuffer;
        Output = output.Buffer as GpuBuffer;

        Rows = input.Rows;
        Columns = input.Columns;
        Value = value;
    }
}