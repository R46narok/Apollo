using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaInvalidPitchValueException : CudaExceptionBase
{
    public CudaInvalidPitchValueException(string message, string file, int line)
        : base(CudaErrorCode.InvalidPitchValue, message, file, line)
    {

    }
}