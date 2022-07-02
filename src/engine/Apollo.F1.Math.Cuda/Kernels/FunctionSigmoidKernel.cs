using System.Runtime.InteropServices;
using Apollo.F1.Math.Cuda.Buffers;
using Apollo.F1.Math.Cuda.Common;

namespace Apollo.F1.Math.Cuda.Kernels;

[KernelEntryPoint("function_sigmoid")]
public class FunctionSigmoidKernel : KernelBase
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "function_sigmoid")]
    private static extern void FunctionSigmoid(IntPtr ptr, int length);
    
    public override void Invoke(GpuBuffer[] buffers)
    {
        EnsureBufferLength(buffers);

        var ptr = buffers[0].Ptr;
        var length = buffers[0].ByteWidth;
        
        FunctionSigmoid(ptr, length);
    }

    private void EnsureBufferLength(GpuBuffer[] buffers)
    {
        if (buffers.Length != 1) throw new ArgumentException();
    }
}
