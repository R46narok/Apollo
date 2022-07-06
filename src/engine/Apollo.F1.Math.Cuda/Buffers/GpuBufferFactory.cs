using Apollo.F1.Math.Common.Buffers;

namespace Apollo.F1.Math.Cuda.Buffers;

public class GpuBufferFactory : IBufferFactory
{
    public IBuffer Allocate(BufferDescriptor descriptor)
    {
        return new GpuBuffer(descriptor);
    }

    public IBuffer TakeOwnership(IntPtr ptr, BufferDescriptor descriptor)
    {
        return new GpuBuffer(ptr, descriptor);
    }
}