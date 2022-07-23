using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaInvalidSymbolException : CudaExceptionBase
{
    public CudaInvalidSymbolException(string message, string file, int line)
        : base(CudaErrorCode.InvalidSymbol, message, file, line)
    {

    }
}