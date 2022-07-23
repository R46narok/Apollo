using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaInsufficientDriverException : CudaExceptionBase
{
    public CudaInsufficientDriverException(string message, string file, int line)
        : base(CudaErrorCode.InsufficientDriver, message, file, line)
    {

    }
}