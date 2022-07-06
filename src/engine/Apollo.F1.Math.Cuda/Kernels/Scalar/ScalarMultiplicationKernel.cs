using System.Runtime.InteropServices;
using Apollo.F1.Math.Cuda.Buffers;
using Apollo.F1.Math.Cuda.Common;

namespace Apollo.F1.Math.Cuda.Kernels;

public class ScalarMultiplicationKernel : KernelBase
{
     [DllImport(Dll.Name, CallingConvention = CallingConvention.Cdecl, EntryPoint = "multiply_scalar")]
     private static extern void MultiplyScalar(IntPtr input, IntPtr output, int length, double scalar);

     private readonly double _scalar;

     public ScalarMultiplicationKernel(double scalar)
     {
         _scalar = scalar;
     }

     public override void Invoke(GpuBuffer[] buffers)
     {
         EnsureBufferLength(buffers);
    
         var input = buffers[0].Ptr;
         var output = buffers[1].Ptr;
         var length = buffers[0].ByteWidth;
         
         MultiplyScalar(input, output, length, _scalar);
     }
    
     private void EnsureBufferLength(GpuBuffer[] buffers)
     {
         if (buffers.Length != 2) throw new ArgumentException();
     }
}
