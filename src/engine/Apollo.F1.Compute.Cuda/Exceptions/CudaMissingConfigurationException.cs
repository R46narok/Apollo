using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaMissingConfigurationException : CudaExceptionBase
{
    public CudaMissingConfigurationException(string message, string file, int line)
        : base(CudaErrorCode.MissingConfiguration, message, file, line)
    {

    }
}
