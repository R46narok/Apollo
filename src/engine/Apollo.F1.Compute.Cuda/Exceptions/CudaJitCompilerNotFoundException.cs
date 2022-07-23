using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaJitCompilerNotFoundException : CudaExceptionBase
{
    public CudaJitCompilerNotFoundException(string message, string file, int line)
        : base(CudaErrorCode.JitCompilerNotFound, message, file, line)
    {

    }
}