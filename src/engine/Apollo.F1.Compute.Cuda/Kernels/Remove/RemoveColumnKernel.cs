using System.Runtime.InteropServices;
using Apollo.F1.Compute.Cuda.Common;
using Apollo.F1.Compute.Cuda.Buffers;

namespace Apollo.F1.Compute.Cuda.Kernels;

[KernelEntryPoint("remove_column")]
public class RemoveColumnKernel : KernelBase<RemoveKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "remove_column")]
    private static extern void RemoveColumn(IntPtr input, IntPtr output, int rows, int columns);

    public override void Invoke(RemoveKernelOptions options)
    {
        var input = options.Input.Ptr;
        var output = options.Output.Ptr;
        var rows = options.Rows;
        var columns = options.Columns;
        
        RemoveColumn(input, output, rows, columns);
    }
}