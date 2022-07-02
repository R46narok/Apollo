using System.Runtime.InteropServices;
using Apollo.F1.Math.Cuda.Buffers;

namespace Apollo.F1.Math.Cuda.Common;

public interface IKernel
{
    public void Invoke(GpuBuffer[] buffers);
}

public class KernelBase : IKernel
{
    public virtual void Invoke(GpuBuffer[] buffers)
    {
        
    }
}