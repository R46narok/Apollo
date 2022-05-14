using System.Security.Claims;
using Apollo.Services.Authorization.Data.Entities;
using Microsoft.AspNetCore.Identity;

namespace Apollo.Services.Authorization.Identity;


public class IdentityService : IIdentityService
{
    private readonly UserManager<ApplicationUser> _userManager;

    public IdentityService(UserManager<ApplicationUser> userManager)
    {
        _userManager = userManager;
    }

    public async Task<ApplicationUser?> FindByNameAsync(string? name)
    {
        return await _userManager.FindByNameAsync(name);
    }

    public Task<IdentityResult> AddClaimAsync(ApplicationUser user, Claim claim)
    {
        return _userManager.AddClaimAsync(user, claim);
    }

    public Task<IList<Claim>> GetClaimsAsync(ApplicationUser user)
    {
        return _userManager.GetClaimsAsync(user);
    }

    public Task<IdentityResult> CreateAsync(ApplicationUser user, string password)
    {
        return _userManager.CreateAsync(user, password);
    }

    public Task<IdentityResult> DeleteAsync(ApplicationUser user)
    {
        return _userManager.DeleteAsync(user);
    }

    public Task<IdentityResult> ChangePasswordAsync(ApplicationUser user, string oldPassword, string newPassword)
    {
        return _userManager.ChangePasswordAsync(user, oldPassword, newPassword);
    }

    public Task<IdentityResult> UpdateAsync(ApplicationUser user)
    {
        return _userManager.UpdateAsync(user);
    }

    public Task<bool> CheckPasswordAsync(ApplicationUser user, string password)
    {
        return _userManager.CheckPasswordAsync(user, password);
    }
}