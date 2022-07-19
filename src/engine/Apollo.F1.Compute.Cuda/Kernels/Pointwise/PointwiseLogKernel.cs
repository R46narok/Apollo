using System.Runtime.InteropServices;
using Apollo.F1.Compute.Cuda.Buffers;
using Apollo.F1.Compute.Cuda.Common;

namespace Apollo.F1.Compute.Cuda.Kernels;

[KernelEntryPoint("pointwise_log")]
public class PointwiseLogKernel : KernelBase<PointwiseOperationKernelOptions>
{
    [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "pointwise_log")]
    private static extern void PointwiseLog(IntPtr input, IntPtr output, int length);
    
    public override void Invoke(PointwiseOperationKernelOptions options)
    {
        var input = options.Operand.Ptr;
        var output = options.Output.Ptr;
        var length = options.Output.ByteWidth / sizeof(double);
            
        PointwiseLog(input, output, length);
    }
}
