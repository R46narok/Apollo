using System;
using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Exceptions;

public class CudaExceptionBase : Exception
{
    public CudaErrorCode Code { get; set; }
    public string File { get; set; }
    public int Line { get; set; }

    public CudaExceptionBase(CudaErrorCode code, string message, string file, int line)
    : base($"[{code}] Message: {message} at file {file}, line {line}")
    {
        Code = code;
        File = file;
        Line = line;
    }
}