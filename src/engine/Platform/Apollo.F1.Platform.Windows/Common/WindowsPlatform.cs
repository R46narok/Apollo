using Apollo.F1.Platform;
using Apollo.F1.Platform.Attributes;
using Apollo.F1.Platform.Common;
using Apollo.F1.Platform.Execution.Interfaces;

namespace Apollo.F1.Platform.Windows.Common;

[PlatformImplementation(PlatformID.Win32NT, typeof(IPlatform))]
public class WindowsPlatform : IPlatform
{
    public string Id => Environment.OSVersion.VersionString;
    
    public IProcess[] GetRunningProcesses()
    {
        throw new NotImplementedException();
    }
}