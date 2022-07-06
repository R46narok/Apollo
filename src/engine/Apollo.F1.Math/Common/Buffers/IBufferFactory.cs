namespace Apollo.F1.Math.Common.Buffers;

public interface IBufferFactory
{
    IBuffer Allocate(BufferDescriptor descriptor);
    IBuffer TakeOwnership(IntPtr ptr, BufferDescriptor descriptor);
}