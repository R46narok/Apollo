using System.Runtime.InteropServices;
using Apollo.F1.Math.Cuda.Buffers;

namespace Apollo.F1.Math.Cuda.Common;

public interface IKernel
{
    public void Invoke(KernelOptionsBase options);
}

public class KernelBase<TOptions> : IKernel
    where TOptions : KernelOptionsBase
{
    public virtual void Invoke(KernelOptionsBase options)
    {
        Invoke((TOptions)options);
    }

    public virtual void Invoke(TOptions options)
    {
        throw new NotImplementedException();
    }
}