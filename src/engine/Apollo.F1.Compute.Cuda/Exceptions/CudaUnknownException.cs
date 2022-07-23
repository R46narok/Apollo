using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaUnknownException : CudaExceptionBase
{
    public CudaUnknownException(string message, string file, int line)
        : base(CudaErrorCode.Unknown, message, file, line)
    {

    }
}