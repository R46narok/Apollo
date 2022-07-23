using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaLaunchTimeoutException : CudaExceptionBase
{
    public CudaLaunchTimeoutException(string message, string file, int line)
        : base(CudaErrorCode.LaunchTimeout, message, file, line)
    {

    }
}
