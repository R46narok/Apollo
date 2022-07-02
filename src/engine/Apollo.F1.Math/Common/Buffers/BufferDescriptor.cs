namespace Apollo.F1.Math.Common.Buffers;

public enum BufferUsage
{
    CpuOnly,
    GpuOnly,
    CpuAndGpu
}

public class BufferDescriptor
{
    public int Offset { get; set; }
    public int Stride { get; set; }
    public int ByteWidth { get; set; }
    public BufferUsage Usage { get; set; }
}