using Microsoft.AspNetCore.Identity;
using Rift.Core.Entities;

namespace Apollo.Services.Authorization.Data.Entities;

public class ApplicationUser : IdentityUser, IEntity<string>
{
    public DateTime CreatedOn { get; set; }
    public DateTime UpdatedOn { get; set; }
}