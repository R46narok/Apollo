using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaInvalidPtxException : CudaExceptionBase
{
    public CudaInvalidPtxException(string message, string file, int line)
        : base(CudaErrorCode.InvalidPtx, message, file, line)
    {

    }
}