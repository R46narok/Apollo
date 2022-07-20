using Apollo.F1.Compute.Common.Buffers;

namespace Apollo.F1.Compute.Cuda.Buffers;

public class GpuBuffer : BufferBase, IDisposable
{
    public GpuBuffer(BufferDescriptor descriptor) : base(descriptor)
    {
        if (descriptor.Offset == 0)
            Ptr = GlobalMemory.Malloc(ByteWidth);
    }

    public GpuBuffer(IntPtr ptr, BufferDescriptor descriptor) : base(descriptor)
    {
        Ptr = ptr;
    }
    
    public override void Upload(double[] data)
    {
        GlobalMemory.CopyHostToDevice(data, Ptr, ByteWidth);
    }

    public override double[]? Read()
    {
        var cpuBuffer = new double[ByteWidth / sizeof(double)];
        GlobalMemory.CopyDeviceToHost(Ptr, cpuBuffer, ByteWidth);

        return cpuBuffer;
    }

    public override void Reset()
    {
        GlobalMemory.Memset(Ptr, ByteWidth, 0);
    }

    public override void Dispose()
    {
        GlobalMemory.Free(Ptr);
    }
}