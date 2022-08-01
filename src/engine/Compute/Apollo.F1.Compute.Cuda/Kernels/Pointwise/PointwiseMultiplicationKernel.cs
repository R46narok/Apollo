using System;
using System.Runtime.InteropServices;
using Apollo.F1.Compute.Cuda.Common;
using Apollo.F1.Compute.Cuda.Buffers;
using Apollo.F1.Compute.Cuda.Common.Execution;
using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Kernels;

public class PointwiseMultiplicationKernel : KernelBase<PointwiseKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pointwise_multiplication")]
    private static extern void PointwiseMultiplication(IntPtr first, IntPtr second, IntPtr output, int length);

    public override void Invoke(PointwiseKernelOptions options)
    {
        var first = options.FirstOperand.Ptr;
        var second = options.SecondOperand.Ptr;
        var output = options.Output.Ptr;
        var length = options.FirstOperand.ByteWidth / sizeof(double);

        PointwiseMultiplication(first, second, output, (int)length);
    }
}