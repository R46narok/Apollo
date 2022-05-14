using Apollo.Services.Authorization.Commands.User.CreateUserCommand;
using Apollo.Services.Authorization.Data.Entities;
using AutoMapper;

namespace Apollo.Services.Authorization.MappingProfiles;

public class UserProfile : Profile
{
    public UserProfile()
    {
        CreateMap<CreateUserCommand, ApplicationUser>();
    }
}