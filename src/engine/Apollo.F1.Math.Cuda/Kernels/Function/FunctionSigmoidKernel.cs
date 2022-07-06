using System.Runtime.InteropServices;
using Apollo.F1.Math.Cuda.Buffers;
using Apollo.F1.Math.Cuda.Common;

namespace Apollo.F1.Math.Cuda.Kernels;


[KernelEntryPoint("function_sigmoid")]
public class FunctionSigmoidKernel : KernelBase<FunctionKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "function_sigmoid")]
    private static extern void FunctionSigmoid(IntPtr ptr, int length);

    public override void Invoke(FunctionKernelOptions options)
    {
        var ptr = options.Output.Ptr;
        var length = options.Output.ByteWidth / sizeof(double);
        
        FunctionSigmoid(ptr, length);
    }
}
