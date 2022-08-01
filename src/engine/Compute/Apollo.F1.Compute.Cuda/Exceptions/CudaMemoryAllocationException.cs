using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaMemoryAllocationException : CudaExceptionBase
{
    public CudaMemoryAllocationException(string message, string file, int line)
        : base(CudaErrorCode.MemoryAllocation, message, file, line)
    {

    }
}