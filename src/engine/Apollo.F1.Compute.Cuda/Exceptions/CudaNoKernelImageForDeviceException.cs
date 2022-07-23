using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaNoKernelImageForDeviceException : CudaExceptionBase
{
    public CudaNoKernelImageForDeviceException(string message, string file, int line)
        : base(CudaErrorCode.NoKernelImageForDevice, message, file, line)
    {

    }
}