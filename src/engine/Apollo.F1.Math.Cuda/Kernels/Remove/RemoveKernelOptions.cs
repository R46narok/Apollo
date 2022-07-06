using Apollo.F1.Math.Common.LinearAlgebra;
using Apollo.F1.Math.Cuda.Buffers;
using Apollo.F1.Math.Cuda.Common;

namespace Apollo.F1.Math.Cuda.Kernels;

public class RemoveKernelOptions : KernelOptionsBase
{
    public GpuBuffer Input { get; set; }
    public int Index { get; set; }
    public int Rows { get; set; }
    public int Columns { get; set; }

    public RemoveKernelOptions(Matrix input, Matrix output, int index)
    {
        Input = input.Buffer as GpuBuffer;
        Output = output.Buffer as GpuBuffer;

        Rows = input.Rows;
        Columns = input.Columns;
        Index = index;
    }
}