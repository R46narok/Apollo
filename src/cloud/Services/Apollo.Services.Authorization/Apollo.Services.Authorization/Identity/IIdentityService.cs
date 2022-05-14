using System.Security.Claims;
using Apollo.Services.Authorization.Data.Entities;
using Microsoft.AspNetCore.Identity;

namespace Apollo.Services.Authorization.Identity;

public interface IIdentityService
{
    public Task<ApplicationUser?> FindByNameAsync(string? userName);
    public Task<IdentityResult> AddClaimAsync(ApplicationUser user, Claim claim);
    public Task<IList<Claim>> GetClaimsAsync(ApplicationUser user);
    public Task<IdentityResult> CreateAsync(ApplicationUser user, string password);
    public Task<IdentityResult> DeleteAsync(ApplicationUser user);
    public Task<IdentityResult> ChangePasswordAsync(ApplicationUser user, string oldPassword, string newPassword);
    public Task<IdentityResult> UpdateAsync(ApplicationUser user);
    public Task<bool> CheckPasswordAsync(ApplicationUser user, string password);
}