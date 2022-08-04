using System.Diagnostics;
using System.Reflection.Metadata;
using System.Text;
using Apollo.F1.Platform.Attributes;
using Apollo.F1.Platform.Execution.Interfaces;
using Apollo.F1.Platform.Windows.Common.Win32.Enums;
using Apollo.F1.Platform.Windows.Common.Win32.Libraries;

namespace Apollo.F1.Platform.Windows.Execution;

[DebuggerDisplay("{Name}, {Id}")]
public class WindowsProcess : IProcess
{
    public DWORD Id { get; private set; }
    public HANDLE Handle { get; private set; }
    public string Name { get; private set; }
    
    public WindowsProcess(DWORD id, HANDLE handle, string name)
    {
        Id = id;
        Handle = handle;
        Name = name;
    }

    public static unsafe WindowsProcess[] EnumerateAll()
    {
        DWORD[] aProcesses = new DWORD[1024];
        DWORD cbNeeded;

        fixed (DWORD* pFirst = &aProcesses[0])
        {
            var result = Psapi.EnumProcesses(new IntPtr(pFirst), sizeof(DWORD) * 1024, new IntPtr(&cbNeeded));
        }

        DWORD cProcesses = cbNeeded / sizeof(DWORD);
        var processes = new WindowsProcess[cProcesses];
        
        for (int i = 0; i < cProcesses; ++i)
        {
            var processId = aProcesses[i];
            HANDLE hProcess = Kernel32.OpenProcess((DWORD) (ProcessAccess.QueryInformation | ProcessAccess.VmRead),
                false,
                processId);
            HMODULE hMod;

            if (hProcess == IntPtr.Zero) continue;

            Psapi.EnumProcessModules(hProcess, &hMod, (uint) sizeof(IntPtr), new IntPtr(&cbNeeded));
            
            var builder = new StringBuilder(32);
            Psapi.GetModuleBaseNameA(hProcess, hMod, builder, 32);

            processes[i] = new WindowsProcess(processId, hProcess, builder.ToString());
        }

        return processes;
    }
}