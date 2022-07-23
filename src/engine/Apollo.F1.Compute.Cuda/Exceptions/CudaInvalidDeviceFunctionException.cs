using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaInvalidDeviceFunctionException : CudaExceptionBase
{
    public CudaInvalidDeviceFunctionException(string message, string file, int line)
        : base(CudaErrorCode.InvalidDeviceFunction, message, file, line)
    {

    }
}