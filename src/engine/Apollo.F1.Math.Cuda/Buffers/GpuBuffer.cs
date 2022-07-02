using Apollo.F1.Math.Common.Buffers;

namespace Apollo.F1.Math.Cuda.Buffers;

public class GpuBuffer : BufferBase, IDisposable
{
    public IntPtr Ptr { get; private set; }

    public GpuBuffer(BufferDescriptor descriptor) : base(descriptor)
    {
        Ptr = Vram.Malloc(ByteWidth);
    }

    public override void Upload(double[] data)
    {
        Vram.CopyHostToDevice(data, Ptr, ByteWidth);
    }

    public override double[]? Read()
    {
        var cpuBuffer = new double[ByteWidth / sizeof(double)];
        Vram.CopyDeviceToHost(Ptr, cpuBuffer, ByteWidth);

        return cpuBuffer;
    }

    public void Dispose()
    {
        Vram.Free(Ptr);
    }
}