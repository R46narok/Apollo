using System.Drawing;
using System.Runtime.InteropServices;
using Apollo.F1.Math.Common.Buffers;
using Apollo.F1.Math.Cuda.Common;

namespace Apollo.F1.Math.Cuda.Kernels;

[KernelEntryPoint("add_scalar")]
public class ScalarAdditionKernel : KernelBase<ScalarKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "add_scalar")]
    private static extern void AddScalar(IntPtr input, IntPtr output, int length, double scalar);
    
    public override void Invoke(ScalarKernelOptions options)
    {
        var inputBuffer = options.Input.Ptr;
        var outputBuffer = options.Output.Ptr;
        var length = options.Output.ByteWidth / sizeof(double);
        var scalar = options.Scalar;

        AddScalar(inputBuffer, outputBuffer, length, scalar);
    }
}