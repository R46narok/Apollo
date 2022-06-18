namespace Apollo.F1.Math.Exceptions;

public class DimensionsException : Exception
{
    public DimensionsException(
        string message = "Unsupported operation on matrices of different dimensions")
        : base(message)
    {
        
    }
}