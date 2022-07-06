using Apollo.F1.Math.Common.Buffers;

namespace Apollo.F1.Math.Cuda.Buffers;

public class GpuBuffer : BufferBase, IDisposable
{
    public GpuBuffer(BufferDescriptor descriptor) : base(descriptor)
    {
        Ptr = Vram.Malloc(ByteWidth);
    }

    public GpuBuffer(IntPtr ptr, BufferDescriptor descriptor) : base(descriptor)
    {
        Ptr = ptr;
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

    public override void Reset()
    {
        Vram.Memset(Ptr, ByteWidth, 0);
    }

    public void Dispose()
    {
        Vram.Free(Ptr);
    }
}