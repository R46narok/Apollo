using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaInvalidResourceHandleException : CudaExceptionBase
{
    public CudaInvalidResourceHandleException(string message, string file, int line)
        : base(CudaErrorCode.InvalidResourceHandle, message, file, line)
    {

    }
}