using System.Runtime.InteropServices;
using Apollo.F1.Compute.Common.LinearAlgebra;
using Apollo.F1.Compute.Cuda.Buffers;
using Apollo.F1.Compute.Cuda.Common;

namespace Apollo.F1.Compute.Cuda.Kernels;

public class MultiplicationKernelOptions : KernelOptionsBase
{
    public GpuBuffer FirstOperand { get; set; }
    public GpuBuffer SecondOperand { get; set; }

    public int FirstRows { get; set; }
    public int FirstColumns { get; set; }
    public int SecondColumns { get; set; }

    public MultiplicationKernelOptions()
    {
    }
    
    public MultiplicationKernelOptions(Matrix first, Matrix second, Matrix output)
    {
        FirstOperand = first.Buffer as GpuBuffer;
        SecondOperand = second.Buffer as GpuBuffer;
        Output = output.Buffer as GpuBuffer;

        FirstRows = first.Rows;
        FirstColumns = first.Columns;
        SecondColumns = second.Columns;
    }
}

[KernelEntryPoint("multiply")]
public class MultiplicationKernel : KernelBase<MultiplicationKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "multiply")]
    private static extern void Multiply(IntPtr first, IntPtr second, IntPtr output, 
        int firstRows, int firstColumns, int secondColumns);

    public override void Invoke(MultiplicationKernelOptions options)
    {
        var first = options.FirstOperand.Ptr;
        var second = options.SecondOperand.Ptr;
        var output = options.Output.Ptr;
        
        var firstRows = options.FirstRows;
        var firstColumns = options.FirstColumns;
        var secondColumns = options.SecondColumns;
        
        Multiply(first, second, output, firstRows, firstColumns, secondColumns);
    }
}
