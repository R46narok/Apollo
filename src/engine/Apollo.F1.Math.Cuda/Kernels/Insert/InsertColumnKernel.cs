using System.Runtime.InteropServices;
using Apollo.F1.Math.Cuda.Buffers;
using Apollo.F1.Math.Cuda.Common;

namespace Apollo.F1.Math.Cuda.Kernels;

[KernelEntryPoint("insert_column")]
public class InsertColumnKernel : KernelBase
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "insert_column")]
    private static extern void InsertColumn(IntPtr src, IntPtr dst, int rows, int columns, double value);

    private readonly int _rows;
    private readonly int _columns;
    private readonly double _value;
    
    public InsertColumnKernel(int rows, int columns, double value)
    {
        _rows = rows;
        _columns = columns;
        _value = value;
    }
    
    public override void Invoke(GpuBuffer[] buffers)
    {
        EnsureBufferLength(buffers);

        var src = buffers[0].Ptr;
        var dst = buffers[1].Ptr;

        InsertColumn(src, dst, _rows, _columns, _value);
    }

    private void EnsureBufferLength(GpuBuffer[] buffers)
    {
        if (buffers.Length != 2) throw new ArgumentException();
    }
}
