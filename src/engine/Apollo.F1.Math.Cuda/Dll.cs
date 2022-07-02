using System.Runtime.InteropServices;

namespace Apollo.F1.Math.Cuda;

public static class Dll
{
   public const string Name = "Apollo.F1.Math.Cuda.Native.dll";
   public static CallingConvention Convention => CallingConvention.Cdecl;
}