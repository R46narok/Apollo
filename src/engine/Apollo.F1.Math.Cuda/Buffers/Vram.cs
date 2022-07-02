// ReSharper disable IdentifierTypo

using System.Runtime.InteropServices;

namespace Apollo.F1.Math.Cuda.Buffers;

public static class Vram
{
    [DllImport(Dll.Name, EntryPoint = "allocate_vram", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr Malloc(int bytes);

    [DllImport(Dll.Name, EntryPoint = "destroy_vram", CallingConvention = CallingConvention.Cdecl)]
    public static extern void Free(IntPtr ptr);

    [DllImport(Dll.Name, EntryPoint = "copy_host_to_device", CallingConvention = CallingConvention.Cdecl)]
    public static extern void CopyHostToDevice([MarshalAs(UnmanagedType.LPArray)] double[] src, IntPtr dst, int length);
    
    [DllImport(Dll.Name, EntryPoint = "copy_device_to_host", CallingConvention = CallingConvention.Cdecl)]
    public static extern void CopyDeviceToHost(IntPtr src, [MarshalAs(UnmanagedType.LPArray)] double[] dst, int length);
        
}