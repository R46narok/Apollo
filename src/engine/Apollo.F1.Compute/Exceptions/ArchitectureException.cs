namespace Apollo.F1.Compute.Exceptions;

public class ArchitectureException : ArgumentException
{
    public ArchitectureException(
        string message = "The given architecture is not valid.") 
        : base(message)
    {
        
    }
}