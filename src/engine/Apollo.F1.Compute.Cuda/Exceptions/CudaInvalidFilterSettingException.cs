using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaInvalidFilterSettingException : CudaExceptionBase
{
    public CudaInvalidFilterSettingException(string message, string file, int line)
        : base(CudaErrorCode.InvalidFilterSetting, message, file, line)
    {

    }
}