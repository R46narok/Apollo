using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaNoDeviceException : CudaExceptionBase
{
    public CudaNoDeviceException(string message, string file, int line)
        : base(CudaErrorCode.NoDevice, message, file, line)
    {

    }
}