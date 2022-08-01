using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaInvalidConfigurationException : CudaExceptionBase
{
    public CudaInvalidConfigurationException(string message, string file, int line)
        : base(CudaErrorCode.InvalidConfiguration, message, file, line)
    {

    }
}