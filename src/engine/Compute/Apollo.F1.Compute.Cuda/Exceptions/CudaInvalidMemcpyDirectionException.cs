using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaInvalidMemcpyDirectionException : CudaExceptionBase
{
    public CudaInvalidMemcpyDirectionException(string message, string file, int line)
        : base(CudaErrorCode.InvalidMemcpyDirection, message, file, line)
    {

    }
}