namespace Apollo.F1.Math.Common.Buffers;

public interface IBuffer
{
    public int Offset { get; }
    public int Stride { get; }
    public int ByteWidth { get; }
    public BufferUsage Usage { get; }

    public void Upload(double[] data);
    public double[]? Read();
    public void Reset();

    public IntPtr Ptr { get; }
}

public class BufferBase : IBuffer
{
     public IntPtr Ptr { get; protected set; }
     public int Offset { get; protected set; }
     public int Stride { get; protected set; }
     public int ByteWidth { get; protected set; }
     public BufferUsage Usage { get; protected set; }
     public virtual void Upload(double[] data)
     {
         
     }

     public virtual void Reset()
     {
         
     }

     public virtual double[]? Read()
     {
         return null;
     }
     
     protected BufferBase(BufferDescriptor descriptor)
     {
         ArgumentNullException.ThrowIfNull(descriptor);
        
         Offset = descriptor.Offset;
         Stride = descriptor.Stride;
         ByteWidth = descriptor.ByteWidth;
         Usage = descriptor.Usage;
     }
}