using Apollo.F1.Compute.Common.Buffers;

namespace Apollo.F1.Compute.Cuda.Buffers;

public class GpuBufferPool : IBufferPool
{
    private readonly IntPtr _start;
    private int _ptr;
    
    public int ByteWidth { get; private set; }

    public GpuBufferPool(int byteWidth)
    {
        ByteWidth = byteWidth;
        _start = GlobalMemory.Malloc(byteWidth);
        _ptr = 0;
    }
    
    public void Dispose()
    {
        GlobalMemory.Free(_start);
    }

    public IBuffer Rent(int bytes)
    {
        if (ByteWidth - _ptr >= bytes)
        {
            var buffer = GlobalMemory.OffsetOf(_start, _ptr);
            var descriptor = new BufferDescriptor(bytes);
            _ptr += bytes;

            return new GpuBuffer(buffer, descriptor);
        }

        return null;
    }
}