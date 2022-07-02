using System.Runtime.InteropServices;
using Apollo.F1.Math.Cuda.Buffers;
using Apollo.F1.Math.Cuda.Common;

namespace Apollo.F1.Math.Cuda.Kernels;

[KernelEntryPoint("function_sigmoid_gradient")]
public class FunctionSigmoidGradientKernel : KernelBase
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "function_sigmoid_gradient")]
    private static extern void FunctionSigmoidGradient(IntPtr ptr, int length);
    
    public override void Invoke(GpuBuffer[] buffers)
    {
        EnsureBufferLength(buffers);

        var ptr = buffers[0].Ptr;
        var length = buffers[0].ByteWidth;
        
        FunctionSigmoidGradient(ptr, length);
    }

    private void EnsureBufferLength(GpuBuffer[] buffers)
    {
        if (buffers.Length != 1) throw new ArgumentException();
    }
}
