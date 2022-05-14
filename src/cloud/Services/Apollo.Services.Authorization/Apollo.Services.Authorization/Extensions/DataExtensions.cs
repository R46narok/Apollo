using System.Net.Http.Headers;
using Apollo.Services.Authorization.Data.Persistence;
using Microsoft.AspNetCore.Builder;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace Apollo.Services.Authorization.Extensions;

public static class DataExtensions
{
    public static void AddPersistence(this WebApplicationBuilder builder, string name = "Database")
    {
        var connectionString = builder.Configuration.GetConnectionString(name);
        builder.Services.AddDbContext<AuthorizationDbContext>(opt =>
            opt.UseSqlServer(connectionString));
    }

    public static void EnsureDatabaseCreated(this WebApplication app)
    {
        using var scope = app.Services.CreateScope();
        var db = scope.ServiceProvider.GetService<AuthorizationDbContext>();
        db!.Database.EnsureCreated();
    }
}