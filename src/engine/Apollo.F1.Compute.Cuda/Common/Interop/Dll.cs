using System.Runtime.InteropServices;

namespace Apollo.F1.Compute.Cuda.Common.Interop;

public static class Dll
{
   public const string Name = "Apollo.F1.Compute.Cuda.Native.dll";
   public static CallingConvention Convention => CallingConvention.Cdecl;
}