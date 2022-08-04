using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using Apollo.F1.Platform.Windows.Common.Win32.Enums;

namespace Apollo.F1.Platform.Windows.Common.Win32.Libraries;

public class Psapi
{
    private const string DllName = "Psapi.dll";

    [DllImport(DllName)]
    public static extern bool EnumProcesses(LPDWORD lpidProcess, DWORD cb, LPDWORD lpcbNeeded);

    [DllImport(DllName)]
    public static extern unsafe bool EnumProcessModules(HANDLE hProcess, HMODULE* lphModule, DWORD cb, LPDWORD lpcbNeeded);

    [DllImport(DllName)]
    public static extern DWORD GetModuleBaseNameA(HANDLE hProcess, HMODULE hModule, StringBuilder lpBaseName, DWORD nSize);
}