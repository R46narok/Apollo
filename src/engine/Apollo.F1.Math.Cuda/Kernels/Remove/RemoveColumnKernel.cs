using System.Runtime.InteropServices;
using Apollo.F1.Math.Cuda.Buffers;
using Apollo.F1.Math.Cuda.Common;

namespace Apollo.F1.Math.Cuda.Kernels;

[KernelEntryPoint("insert_column")]
public class RemoveColumnKernel : KernelBase
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "remove_column")]
    private static extern void RemoveColumn(IntPtr src, IntPtr dst, int rows, int columns);

    private readonly int _rows;
    private readonly int _columns;
    
    public RemoveColumnKernel(int rows, int columns)
    {
        _rows = rows;
        _columns = columns;
    }
    
    public override void Invoke(GpuBuffer[] buffers)
    {
        EnsureBufferLength(buffers);

        var src = buffers[0].Ptr;
        var dst = buffers[1].Ptr;

        RemoveColumn(src, dst, _rows, _columns);
    }

    private void EnsureBufferLength(GpuBuffer[] buffers)
    {
        if (buffers.Length != 2) throw new ArgumentException();
    }
}