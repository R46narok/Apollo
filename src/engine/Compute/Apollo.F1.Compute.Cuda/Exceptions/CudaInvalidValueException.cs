using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaInvalidValueException : CudaExceptionBase
{
    public CudaInvalidValueException(string message, string file, int line)
        : base(CudaErrorCode.InvalidValue, message, file, line)
    {

    }
}