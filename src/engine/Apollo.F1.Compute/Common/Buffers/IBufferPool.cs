namespace Apollo.F1.Compute.Common.Buffers;

public interface IBufferPool : IDisposable
{
    public int ByteWidth { get; }

    public IBuffer Rent(int bytes);
}