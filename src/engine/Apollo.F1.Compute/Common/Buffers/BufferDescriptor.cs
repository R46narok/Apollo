namespace Apollo.F1.Compute.Common.Buffers;

public enum BufferUsage
{
    CpuOnly,
    GpuOnly,
}

public class BufferDescriptor
{
    public int Offset { get; set; }
    public int Stride { get; set; }
    public int ByteWidth { get; set; }
    public BufferUsage Usage { get; set; }

    public BufferDescriptor()
    {
        
    }

    public BufferDescriptor(int byteWidth)
    {
        ByteWidth = byteWidth;
    }
}