using System.Runtime.InteropServices;
using Apollo.F1.Math.Cuda.Buffers;
using Apollo.F1.Math.Cuda.Common;

namespace Apollo.F1.Math.Cuda.Kernels;


[KernelEntryPoint("insert_row")]
public class InsertRowKernel : KernelBase<InsertKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "insert_row")]
    private static extern void InsertRow(IntPtr input, IntPtr output, int rows, int columns, double value);

    public override void Invoke(InsertKernelOptions options)
    {
        var input = options.Input.Ptr;
        var output = options.Output.Ptr;
        var rows = options.Rows;
        var columns = options.Columns;
        var value = options.Value;
        
        InsertRow(input, output, rows, columns, value);
    }
}
