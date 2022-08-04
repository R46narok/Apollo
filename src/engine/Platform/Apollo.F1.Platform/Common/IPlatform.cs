using Apollo.F1.Platform.Execution.Interfaces;

namespace Apollo.F1.Platform.Common;

public interface IPlatform
{
    public string Id { get; }

    public IProcess[] GetRunningProcesses();
}