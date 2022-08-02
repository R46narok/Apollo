namespace Apollo.F1.Platform.Attributes;

[AttributeUsage(AttributeTargets.Class | AttributeTargets.Struct, Inherited = false, AllowMultiple = false)]
public class PlatformImplementationAttribute : Attribute
{
    public PlatformID Id { get; }
    public Type InterfaceType { get; }

    public PlatformImplementationAttribute(PlatformID id, Type interfaceType)
    {
        Id = id;
        InterfaceType = interfaceType;
    }
}