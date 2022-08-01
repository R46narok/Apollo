using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaInvalidDeviceException : CudaExceptionBase
{
    public CudaInvalidDeviceException(string message, string file, int line)
        : base(CudaErrorCode.InvalidDevice, message, file, line)
    {

    }
}