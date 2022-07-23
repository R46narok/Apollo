using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaJitCompilationDisabledException : CudaExceptionBase
{
    public CudaJitCompilationDisabledException(string message, string file, int line)
        : base(CudaErrorCode.JitCompilationDisabled, message, file, line)
    {

    }
}