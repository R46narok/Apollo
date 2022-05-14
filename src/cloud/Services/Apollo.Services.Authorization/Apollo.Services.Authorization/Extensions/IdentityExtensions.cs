using Apollo.Services.Authorization.Data.Entities;
using Apollo.Services.Authorization.Data.Persistence;
using Microsoft.AspNetCore.Identity;
using Microsoft.Extensions.DependencyInjection;

namespace Apollo.Services.Authorization.Extensions;

public static class IdentityExtensions
{
    public static void AddIdentity(this IServiceCollection services)
    {
        services.AddIdentity<ApplicationUser, IdentityRole>()
            .AddEntityFrameworkStores<AuthorizationDbContext>()
            .AddDefaultTokenProviders();
    }
}