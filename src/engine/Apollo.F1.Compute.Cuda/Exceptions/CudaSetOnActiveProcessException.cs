using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaSetOnActiveProcessException : CudaExceptionBase
{
    public CudaSetOnActiveProcessException(string message, string file, int line)
        : base(CudaErrorCode.SetOnActiveProcess, message, file, line)
    {

    }
}