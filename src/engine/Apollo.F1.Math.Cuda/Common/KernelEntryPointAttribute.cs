namespace Apollo.F1.Math.Cuda.Common;

[AttributeUsage(AttributeTargets.Class, Inherited = false, AllowMultiple = false)]
public class KernelEntryPointAttribute : Attribute
{
    public string Function { get; private set; }
    
    public KernelEntryPointAttribute(string function)
    {
        Function = function;
    }
}