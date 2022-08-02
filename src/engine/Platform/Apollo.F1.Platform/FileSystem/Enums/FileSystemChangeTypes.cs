namespace Apollo.F1.Platform.FileSystem.Enums;

[Flags]
public enum FileSystemChangeTypes
{
  Created = 1,
  Deleted = 2,
  Changed = 4,
  Renamed = 8,
  All = Renamed | Changed | Deleted | Created, // 0x0000000F
}
