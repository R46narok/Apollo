namespace Apollo.F1.Compute.Common.Buffers;

public record class BufferBatchElement(int ByteWidth, BufferDataType DataType, string Name);

public class BufferBatch : IDisposable
{
    private readonly Dictionary<string, IBuffer> _buffers = new ();
    private readonly IBufferAllocator _allocator;
    private IBuffer _memoryPool = null!;
    
    public IBuffer this[string key] => _buffers[key];
    
    public BufferBatch(IBufferAllocator allocator, BufferBatchElement[] elements)
    {
        _allocator = allocator;
        InitializeMemoryPool(elements);
        InitializeBuffers(elements);
    }

    private void InitializeBuffers(BufferBatchElement[] elements)
    {
        int offset = 0;
        foreach (var element in elements)
        {
            if (_buffers.ContainsKey(element.Name)) throw new ArgumentException();
            var descriptor = new BufferDescriptor(element.ByteWidth, offset);
            var buffer = _allocator.TakeOwnership(_memoryPool.Ptr, descriptor);
            
            _buffers.Add(element.Name, buffer);
            offset += element.ByteWidth;
        }
    }

    private void InitializeMemoryPool(BufferBatchElement[] elements)
    {
        long byteWidth = 0;
        foreach (var element in elements)
        {
            byteWidth += element.ByteWidth;
        }

        var descriptor = new BufferDescriptor(byteWidth);
        _memoryPool = _allocator.Allocate(descriptor);
    }
    
    public void Dispose()
    {
        _memoryPool.Dispose();
    }
}