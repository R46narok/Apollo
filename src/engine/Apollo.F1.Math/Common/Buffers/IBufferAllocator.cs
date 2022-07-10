namespace Apollo.F1.Math.Common.Buffers;

public interface IBufferAllocator
{
    IBuffer Allocate(BufferDescriptor descriptor);
    void Deallocate(IBuffer buffer);
    
    IBuffer TakeOwnership(IntPtr ptr, BufferDescriptor descriptor);
}