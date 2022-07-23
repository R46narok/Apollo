using System.Runtime.InteropServices;
using Apollo.F1.Compute.Cuda.Buffers;
using Apollo.F1.Compute.Cuda.Common;
using Apollo.F1.Compute.Cuda.Common.Execution;
using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Kernels;


[KernelEntryPoint("multiply_scalar")]
public class ScalarMultiplicationKernel : KernelBase<ScalarKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "multiply_scalar")]
    private static extern void MultiplyScalar(IntPtr input, IntPtr output, int length, double scalar);
    
    public override void Invoke(ScalarKernelOptions options)
    {
        var inputBuffer = options.Input.Ptr;
        var outputBuffer = options.Output.Ptr;
        var length = options.Output.ByteWidth / sizeof(double);
        var scalar = options.Scalar;

        MultiplyScalar(inputBuffer, outputBuffer, (int)length, scalar);
    }
}
