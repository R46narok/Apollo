using System.Runtime.InteropServices;
using Apollo.F1.Math.Cuda.Buffers;
using Apollo.F1.Math.Cuda.Common;

namespace Apollo.F1.Math.Cuda.Kernels;

public class PointwiseSubtractionKernel : KernelBase
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pointwise_subtraction")]
    private static extern void PointwiseSubtraction(IntPtr first, IntPtr second, IntPtr output, int length);
    
    public override void Invoke(GpuBuffer[] buffers)
    {
        EnsureBufferLength(buffers);

        var first = buffers[0].Ptr;
        var second = buffers[1].Ptr;
        var output = buffers[2].Ptr;
        var length = buffers[0].ByteWidth;
        
        PointwiseSubtraction(first, second, output, length);
    }

    private void EnsureBufferLength(GpuBuffer[] buffers)
    {
        if (buffers.Length != 3) throw new ArgumentException();
    }
}