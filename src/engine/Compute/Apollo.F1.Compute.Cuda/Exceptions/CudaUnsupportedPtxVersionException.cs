using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaUnsupportedPtxVersionException : CudaExceptionBase
{
    public CudaUnsupportedPtxVersionException(string message, string file, int line)
        : base(CudaErrorCode.UnsupportedPtxVersion, message, file, line)
    {

    }
}