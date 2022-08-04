using System.Runtime.InteropServices;

namespace Apollo.F1.Platform.Windows.Common.Win32.Libraries;

public static class Kernel32
{
    private const string DllName = "Kernel32.dll";
    
    [DllImport(DllName)]
    public static extern HANDLE OpenProcess(DWORD dwDesiredAccess, bool bInheritHandle, DWORD dwProcessId);

    [DllImport(DllName)]
    public static extern bool CloseHandle(HANDLE hObject);
}