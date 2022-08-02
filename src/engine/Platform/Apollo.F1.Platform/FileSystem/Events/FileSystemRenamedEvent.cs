using Apollo.F1.Platform.FileSystem.Enums;

namespace Apollo.F1.Platform.FileSystem.Events;

public class FileSystemRenamedEvent : FileSystemEventBase
{
    public string OldFullPath { get; set; }
    
    public FileSystemRenamedEvent(string oldFullPath, string fullPath, string name, FileSystemChangeTypes changeType)
        : base(fullPath, name, changeType)
    {
        OldFullPath = oldFullPath;
    }
}