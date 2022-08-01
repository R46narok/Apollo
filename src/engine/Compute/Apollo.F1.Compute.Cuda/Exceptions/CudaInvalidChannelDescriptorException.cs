using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaInvalidChannelDescriptorException : CudaExceptionBase
{
    public CudaInvalidChannelDescriptorException(string message, string file, int line)
        : base(CudaErrorCode.InvalidChannelDescriptor, message, file, line)
    {

    }
}