namespace Apollo.F1.Platform.Windows.Common.Win32.Enums;

[Flags]
public enum ProcessAccess
{
    QueryInformation = 0x0400,
    VmRead = 0x0010
}