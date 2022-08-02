using System.Reflection;
using Apollo.F1.Platform.Attributes;
using Microsoft.Extensions.DependencyInjection;

namespace Apollo.F1.Platform.Extensions;

public static class DependencyInjectionExtensions
{
    public static IServiceCollection AddPlatform(this IServiceCollection services, Assembly[] assemblies)
    {
        if (assemblies.Length == 0) throw new ArgumentException();

        var platformId = Environment.OSVersion.Platform;
        foreach (var assembly in assemblies)
        {
            var pairs = ScanAssembly(assembly, platformId);
            foreach (var pair in pairs)
            {
                services.AddTransient(pair.Key, pair.Value);
            }
        }
        
        return services;
    }

    private static Dictionary<Type, Type> ScanAssembly(Assembly assembly, PlatformID platformId)
    {
        var types = assembly
            .GetTypes()
            .Where(x => x.GetCustomAttribute<PlatformImplementationAttribute>() is not null)
            .ToArray();

        var dictionary = new Dictionary<Type, Type>();
        
        foreach (var type in types)
        {
            var attribute = type.GetCustomAttribute<PlatformImplementationAttribute>()!;
            if (attribute.Id == platformId)
            {
                dictionary.Add(attribute.InterfaceType, type);
            }
        }

        return dictionary;
    }
}