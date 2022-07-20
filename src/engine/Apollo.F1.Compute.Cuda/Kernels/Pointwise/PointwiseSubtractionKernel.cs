using System.Runtime.InteropServices;
using Apollo.F1.Compute.Cuda.Common;
using Apollo.F1.Compute.Cuda.Buffers;

namespace Apollo.F1.Compute.Cuda.Kernels;

public class PointwiseSubtractionKernel : KernelBase<PointwiseKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pointwise_subtraction")]
    private static extern void PointwiseSubtraction(IntPtr first, IntPtr second, IntPtr output, int length);
    
    public override void Invoke(PointwiseKernelOptions options)
    {
        var first = options.FirstOperand.Ptr;
        var second = options.SecondOperand.Ptr;
        var output = options.Output.Ptr;
        var length = options.FirstOperand.ByteWidth / sizeof(double);

        PointwiseSubtraction(first, second, output, (int)length);
    }
}