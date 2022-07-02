using System.Runtime.InteropServices;
using Apollo.F1.Math.Cuda.Buffers;
using Apollo.F1.Math.Cuda.Common;

namespace Apollo.F1.Math.Cuda.Kernels;

public class TransposeKernel : KernelBase
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "transpose")]
    private static extern void Transpose(IntPtr input, IntPtr output, int rows, int columns);

    private readonly int _rows;
    private readonly int _columns;
    
    public TransposeKernel(int rows, int columns)
    {
        _rows = rows;
        _columns = columns;
    }
    
    public override void Invoke(GpuBuffer[] buffers)
    {
        EnsureBufferLength(buffers);

        var input = buffers[0].Ptr;
        var output = buffers[1].Ptr;
        
        Transpose(input, output, _rows, _columns);
    }

    private void EnsureBufferLength(GpuBuffer[] buffers)
    {
        if (buffers.Length != 2) throw new ArgumentException();
    }
}
