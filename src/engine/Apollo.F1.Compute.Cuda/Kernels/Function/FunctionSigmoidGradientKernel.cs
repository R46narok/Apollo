using System.Runtime.InteropServices;
using Apollo.F1.Compute.Cuda.Buffers;
using Apollo.F1.Compute.Cuda.Common;
using Apollo.F1.Compute.Cuda.Common.Execution;
using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Kernels;

[KernelEntryPoint("function_sigmoid_gradient")]
public class FunctionSigmoidGradientKernel : KernelBase<FunctionKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "function_sigmoid_gradient")]
    private static extern void FunctionSigmoidGradient(IntPtr input, IntPtr output, int length);

    public override void Invoke(FunctionKernelOptions options)
    {
        var input = options.Input.Ptr;
        var output = options.Output.Ptr;
        var length = options.Output.ByteWidth / sizeof(double);
        
        FunctionSigmoidGradient(input, output, (int)length);
    }
}
