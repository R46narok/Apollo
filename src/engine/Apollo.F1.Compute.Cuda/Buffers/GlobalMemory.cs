// ReSharper disable IdentifierTypo

using System.Runtime.InteropServices;
using Apollo.F1.Compute.Cuda.Common;
using Apollo.F1.Compute.Cuda.Common.Interop;

namespace Apollo.F1.Compute.Cuda.Buffers;

public static class GlobalMemory
{
    [DllImport(Dll.Name, EntryPoint = "allocate_global_memory", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr Malloc(long bytes);

    [DllImport(Dll.Name, EntryPoint = "destroy_global_memory", CallingConvention = CallingConvention.Cdecl)]
    public static extern void Free(IntPtr ptr);

    [DllImport(Dll.Name, EntryPoint = "copy_host_to_device", CallingConvention = CallingConvention.Cdecl)]
    public static extern void CopyHostToDevice([MarshalAs(UnmanagedType.LPArray)] double[] src, IntPtr dst, long length);
    
    [DllImport(Dll.Name, EntryPoint = "copy_device_to_host", CallingConvention = CallingConvention.Cdecl)]
    public static extern void CopyDeviceToHost(IntPtr src, [MarshalAs(UnmanagedType.LPArray)] double[] dst, long length);

    [DllImport(Dll.Name, EntryPoint = "copy_device_to_device", CallingConvention = CallingConvention.Cdecl)]
    public static extern void CopyDeviceToDevice(IntPtr src, IntPtr dst, long length);

    [DllImport(Dll.Name, EntryPoint = "device_memset", CallingConvention = CallingConvention.Cdecl)]
    public static extern void Memset(IntPtr dst, long length, int value);
    
    public static IntPtr OffsetOf(IntPtr ptr, int offset)
    {
        return ptr + offset;
    }
}