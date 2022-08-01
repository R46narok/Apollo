using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaStartupFailureException : CudaExceptionBase
{
    public CudaStartupFailureException(string message, string file, int line)
        : base(CudaErrorCode.StartupFailure, message, file, line)
    {

    }
}