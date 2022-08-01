using System;
using System.Runtime.InteropServices;
using Apollo.F1.Compute.Cuda.Common;
using Apollo.F1.Compute.Cuda.Common.Execution;
using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Kernels;

public class PointwiseScaledSubtractionKernel : KernelBase<PointwiseKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pointwise_scaled_subtraction")]
    private static extern void PointwiseScaledSubtraction(IntPtr first, IntPtr second, IntPtr output, int length, double scale);
    
    public override void Invoke(PointwiseKernelOptions options)
    {
        var first = options.FirstOperand.Ptr;
        var second = options.SecondOperand.Ptr;
        var output = options.Output.Ptr;
        var scale = options.Scale;
        var length = options.FirstOperand.ByteWidth / sizeof(double);
        
        PointwiseScaledSubtraction(first, second, output, (int)length, scale);
    }
}
