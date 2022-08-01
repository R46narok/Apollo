using System;

namespace Apollo.F1.Compute.Common.Buffers;

public interface IBuffer : IDisposable
{
    public int Offset { get; }
    public long ByteWidth { get; }

    public void Upload(double[] data);
    public double[]? Read();
    public void Reset();

    public IntPtr Ptr { get; }
}