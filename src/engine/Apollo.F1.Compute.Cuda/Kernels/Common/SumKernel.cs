using System.Runtime.InteropServices;
using Apollo.F1.Compute.Common.LinearAlgebra;
using Apollo.F1.Compute.Cuda.Buffers;
using Apollo.F1.Compute.Cuda.Common;

namespace Apollo.F1.Compute.Cuda.Kernels;

public class SumKernelOptions : KernelOptionsBase
{
    public GpuBuffer Input { get; set; }
    public int Rows { get; set; }
    public int Columns { get; set; }

    public SumKernelOptions(MatrixStorage input, GpuBuffer output)
    {
        Input = input.Buffer as GpuBuffer;
        Rows = input.Rows;
        Columns = input.Columns;
        Output = output;
    }
}

public class SumKernel : KernelBase<SumKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "sum")]
    private static extern void Sum(IntPtr input, IntPtr output, int rows, int columns);

    public override void Invoke(SumKernelOptions options)
    {
        var input = options.Input.Ptr;
        var output = options.Output.Ptr;
        var rows = options.Rows;
        var columns = options.Columns;
        
        Sum(input, output, rows, columns);
    }
}
