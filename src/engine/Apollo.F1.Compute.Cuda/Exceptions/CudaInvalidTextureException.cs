using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaInvalidTextureException : CudaExceptionBase
{
    public CudaInvalidTextureException(string message, string file, int line)
        : base(CudaErrorCode.InvalidTexture, message, file, line)
    {

    }
}