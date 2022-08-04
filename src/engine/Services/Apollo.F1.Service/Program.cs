using System.Diagnostics;
using Apollo.F1.Platform.Extensions;
using Apollo.F1.Platform.FileSystem.Interfaces;
using Apollo.F1.Platform.Windows.Common;
using Apollo.F1.Platform.Windows.Common.Win32.Libraries;
using Apollo.F1.Platform.Windows.Execution;

var builder = Host.CreateDefaultBuilder(args);
builder.ConfigureServices(services =>
{
    services.AddPlatform(new []{ typeof(WindowsPlatform).Assembly });
});
var app = builder.Build();

var processes = WindowsProcess.EnumerateAll();
using var scope = app.Services.CreateScope();
var fs = scope.ServiceProvider.GetService<IFileSystem>();



