using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaLaunchFailureException : CudaExceptionBase
{
    public CudaLaunchFailureException(string message, string file, int line)
        : base(CudaErrorCode.LaunchFailure, message, file, line)
    {

    }
}
