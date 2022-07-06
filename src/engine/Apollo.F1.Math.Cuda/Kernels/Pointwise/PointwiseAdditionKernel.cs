using System.Runtime.InteropServices;
using Apollo.F1.Math.Cuda.Buffers;
using Apollo.F1.Math.Cuda.Common;

namespace Apollo.F1.Math.Cuda.Kernels;

[KernelEntryPoint("pointwise_addition")]
public class PointwiseAdditionKernel : KernelBase<PointwiseKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pointwise_addition")]
    private static extern void PointwiseAddition(IntPtr first, IntPtr second, IntPtr output, int length);

    public override void Invoke(PointwiseKernelOptions options)
    {
        var first = options.FirstOperand.Ptr;
        var second = options.SecondOperand.Ptr;
        var output = options.Output.Ptr;
        var length = options.FirstOperand.ByteWidth / sizeof(double);
        
        PointwiseAddition(first, second, output, length);
    }
}