using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaInvalidNormSettingException : CudaExceptionBase
{
    public CudaInvalidNormSettingException(string message, string file, int line)
        : base(CudaErrorCode.InvalidNormSetting, message, file, line)
    {

    }
}