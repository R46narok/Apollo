using System.Runtime.InteropServices;
using Apollo.F1.Math.Cuda.Buffers;
using Apollo.F1.Math.Cuda.Common;

namespace Apollo.F1.Math.Cuda.Kernels;

[KernelEntryPoint("pointwise_addition")]
public class PointwiseLogKernel : KernelBase
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pointwise_log")]
    private static extern void PointwiseLog(IntPtr input, IntPtr output, int length);
    
    public override void Invoke(GpuBuffer[] buffers)
    {
        EnsureBufferLength(buffers);

        var input = buffers[0].Ptr;
        var output = buffers[1].Ptr;
        var length = buffers[0].ByteWidth;
        
        PointwiseLog(input, output, length);
    }

    private void EnsureBufferLength(GpuBuffer[] buffers)
    {
        if (buffers.Length != 2) throw new ArgumentException();
    }
}
