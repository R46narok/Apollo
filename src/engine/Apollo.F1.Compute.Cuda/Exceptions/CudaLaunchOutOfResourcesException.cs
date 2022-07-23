using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaLaunchOutOfResourcesException : CudaExceptionBase
{
    public CudaLaunchOutOfResourcesException(string message, string file, int line)
        : base(CudaErrorCode.LaunchOutOfResources, message, file, line)
    {

    }
}
