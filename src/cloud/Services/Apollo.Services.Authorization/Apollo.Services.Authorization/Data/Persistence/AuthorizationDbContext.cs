using Apollo.Services.Authorization.Data.Entities;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;

namespace Apollo.Services.Authorization.Data.Persistence;

public class AuthorizationDbContext : IdentityDbContext<ApplicationUser>
{
    public AuthorizationDbContext()
    {
        
    }

    public AuthorizationDbContext(DbContextOptions<AuthorizationDbContext> options) : base(options)
    {
        
    }
}