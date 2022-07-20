using System.Runtime.InteropServices;
using Apollo.F1.Compute.Cuda.Common;

namespace Apollo.F1.Compute.Cuda.Kernels;

[KernelEntryPoint("subtract_scalar")]
public class ScalarSubtractionKernel : KernelBase<ScalarKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "subtract_scalar")]
    private static extern void SubtractScalar(IntPtr input, IntPtr output, int length, double scalar);
    
    public override void Invoke(ScalarKernelOptions options)
    {
        var inputBuffer = options.Input.Ptr;
        var outputBuffer = options.Output.Ptr;
        var length = options.Output.ByteWidth / sizeof(double);
        var scalar = options.Scalar;

        SubtractScalar(inputBuffer, outputBuffer, (int)length, scalar);
    }
}
