using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaUnmapBufferObjectFailedException : CudaExceptionBase
{
    public CudaUnmapBufferObjectFailedException(string message, string file, int line)
        : base(CudaErrorCode.UnmapBufferObjectFailed, message, file, line)
    {

    }
}