using Apollo.Services.Authorization.Commands.Token.CreateTokenCommand;
using Apollo.Services.Authorization.DTOs;
using AutoMapper;

namespace Apollo.Services.Authorization.MappingProfiles;


public class TokenProfile : Profile
{
    public TokenProfile()
    {
        CreateMap<UserCredentialsDto, CreateTokenCommand>();
    }
}