using Apollo.F1.Platform.Attributes;
using Apollo.F1.Platform.Execution.Interfaces;

namespace Apollo.F1.Platform.Windows.Execution;

[PlatformImplementation(PlatformID.Win32NT, typeof(IThread))]
public class WindowsThread : IThread
{
    
}