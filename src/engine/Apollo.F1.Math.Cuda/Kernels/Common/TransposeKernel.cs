using System.Runtime.InteropServices;
using Apollo.F1.Math.Common.LinearAlgebra;
using Apollo.F1.Math.Cuda.Buffers;
using Apollo.F1.Math.Cuda.Common;

namespace Apollo.F1.Math.Cuda.Kernels;

public class TransposeKernelOptions : KernelOptionsBase
{
    public GpuBuffer Input { get; set; }
    public int Rows { get; set; }
    public int Columns { get; set; }

    public TransposeKernelOptions(Matrix input, Matrix output)
    {
        Input = input.Buffer as GpuBuffer;
        Output = output.Buffer as GpuBuffer;

        Rows = input.Rows;
        Columns = input.Columns;
    }
}

public class TransposeKernel : KernelBase<TransposeKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "transpose")]
    private static extern void Transpose(IntPtr input, IntPtr output, int rows, int columns);

    public override void Invoke(TransposeKernelOptions options)
    {
        var input = options.Input.Ptr;
        var output = options.Output.Ptr;
        var rows = options.Rows;
        var columns = options.Columns;
        
        Transpose(input, output, rows, columns);
    }
}
