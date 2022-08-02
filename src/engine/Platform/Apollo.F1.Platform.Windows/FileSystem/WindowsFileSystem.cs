using Apollo.F1.Platform.Attributes;
using Apollo.F1.Platform.FileSystem.Enums;
using Apollo.F1.Platform.FileSystem.Events;
using Apollo.F1.Platform.FileSystem.Interfaces;
using Apollo.F1.Platform.Interfaces;

namespace Apollo.F1.Platform.Windows.FileSystem;

[PlatformImplementation(PlatformID.Win32NT, typeof(IFileSystem))]
public class WindowsFileSystem : IFileSystem
{
    private readonly FileSystemWatcher _watcher;

    public event EventHandler<FileSystemEventBase>? Created;
    public event EventHandler<FileSystemEventBase>? Changed;
    public event EventHandler<FileSystemEventBase>? Deleted;
    public event EventHandler<FileSystemRenamedEvent>? Renamed;
    
    public WindowsFileSystem(string path, string filter = "*.*")
    {
        if (string.IsNullOrEmpty(path)) throw new ArgumentException();
        
        _watcher = new FileSystemWatcher(path);
        _watcher.NotifyFilter =  NotifyFilters.Attributes
                               | NotifyFilters.CreationTime
                               | NotifyFilters.DirectoryName
                               | NotifyFilters.FileName
                               | NotifyFilters.LastAccess
                               | NotifyFilters.LastWrite
                               | NotifyFilters.Security
                               | NotifyFilters.Size;

        InitializeEvents();

        _watcher.Filter = filter;
        _watcher.IncludeSubdirectories = true;
        _watcher.EnableRaisingEvents = true;
    }

    private void InitializeEvents()
    {
        _watcher.Created += (_, args) => Created?.Invoke(this, ConvertToFileSystemEventBase(args));
        _watcher.Changed += (_, args) => Changed?.Invoke(this, ConvertToFileSystemEventBase(args));
        _watcher.Deleted += (_, args) => Deleted?.Invoke(this, ConvertToFileSystemEventBase(args));
        _watcher.Renamed += (_, args) => Renamed?.Invoke(this, ConvertToFileSystemEventBase(args));
    }

    private FileSystemEventBase ConvertToFileSystemEventBase(FileSystemEventArgs args)
    {
        return new FileSystemEventBase(args.FullPath, args.Name, (FileSystemChangeTypes) args.ChangeType);
    }
    
    private FileSystemRenamedEvent ConvertToFileSystemEventBase(RenamedEventArgs args)
    {
        return new FileSystemRenamedEvent(args.OldFullPath, args.FullPath, args.Name, (FileSystemChangeTypes) args.ChangeType);
    }
}