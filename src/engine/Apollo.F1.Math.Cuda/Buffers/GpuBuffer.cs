using Apollo.F1.Math.Common.Buffers;

namespace Apollo.F1.Math.Cuda.Buffers;

public class GpuBuffer : BufferBase, IDisposable
{
    public IntPtr Ptr { get; private set; }

    public GpuBuffer(BufferDescriptor descriptor) : base(descriptor)
    {
        Ptr = Vram.Malloc(ByteWidth);
    }

    public void Dispose()
    {
        Vram.Free(Ptr);
    }
}