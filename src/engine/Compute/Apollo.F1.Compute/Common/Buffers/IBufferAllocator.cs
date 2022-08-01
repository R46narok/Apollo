using System;

namespace Apollo.F1.Compute.Common.Buffers;

public interface IBufferAllocator
{
    IBuffer Allocate(BufferDescriptor descriptor);
    void Deallocate(IBuffer buffer);
    
    IBuffer TakeOwnership(IntPtr ptr, BufferDescriptor descriptor);
}