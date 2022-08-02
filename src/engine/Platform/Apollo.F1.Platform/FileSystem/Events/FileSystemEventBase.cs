using Apollo.F1.Platform.FileSystem.Enums;

namespace Apollo.F1.Platform.FileSystem.Events;

public class FileSystemEventBase : EventArgs
{
    public string FullPath { get; set; }
    public string Name { get; set; }
    public FileSystemChangeTypes ChangeType { get; set; }

    public FileSystemEventBase(string fullPath, string name, FileSystemChangeTypes changeType)
    {
        FullPath = fullPath;
        Name = name;
        ChangeType = changeType;
    }
}