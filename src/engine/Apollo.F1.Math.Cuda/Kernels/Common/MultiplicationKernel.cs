using System.Runtime.InteropServices;
using Apollo.F1.Math.Cuda.Buffers;
using Apollo.F1.Math.Cuda.Common;

namespace Apollo.F1.Math.Cuda.Kernels;

[KernelEntryPoint("multiply")]
public class MultiplicationKernel : KernelBase
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "multiply")]
    private static extern void Multiply(IntPtr first, IntPtr second, IntPtr output, int m, int n, int k);

    private readonly int _m;
    private readonly int _n;
    private readonly int _k;
    
    public MultiplicationKernel(int m, int n, int k)
    {
        _m = m;
        _n = n;
        _k = k;
    }
    
    public override void Invoke(GpuBuffer[] buffers)
    {
        EnsureBufferLength(buffers);

        var first = buffers[0].Ptr;
        var second = buffers[1].Ptr;
        var output = buffers[2].Ptr;

        Multiply(first, second, output, _m, _n, _k);
    }

    private void EnsureBufferLength(GpuBuffer[] buffers)
    {
        if (buffers.Length != 3) throw new ArgumentException();
    }
}
