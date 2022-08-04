using Apollo.F1.Platform.Attributes;
using Apollo.F1.Platform.Network.Interfaces;

namespace Apollo.F1.Platform.Windows.Network;

[PlatformImplementation(PlatformID.Win32NT, typeof(INetwork))]
public class WindowsNetwork : INetwork
{
    
}