using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaInvalidTextureBindingException : CudaExceptionBase
{
    public CudaInvalidTextureBindingException(string message, string file, int line)
        : base(CudaErrorCode.InvalidTextureBinding, message, file, line)
    {

    }
}