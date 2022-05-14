using Apollo.Services.Authorization.Authorization;
using Apollo.Services.Authorization.Data.Entities;
using Apollo.Services.Authorization.Identity;
using AutoMapper;
using MediatR;
using Microsoft.Extensions.Configuration;
using Rift.CloudProviders.Common;
using Rift.CloudProviders.MessageBrokers;
using Rift.Core.Types;

namespace Apollo.Services.Authorization.Commands.User.CreateUserCommand;

#nullable disable
public class CreateUserCommand : IRequest<ApiResponse>
{
    public string UserName { get; set; }
    public string Email { get; set; }
    public string Password { get; set; }

    public CreateUserCommand()
    {
        
    }
    
    public CreateUserCommand(string userName, string email, string password)
    {
        UserName = userName;
        Email = email;
        Password = password;
    }
}

public class CreateUserCommandHandler : IRequestHandler<CreateUserCommand, ApiResponse>
{
    private readonly IMapper _mapper;
    private readonly IIdentityService _identityService;
    private readonly ICloudServiceProvider _cloudServiceProvider;
    private readonly IConfiguration _configuration;

    public CreateUserCommandHandler(IMapper mapper, IIdentityService identityService, ICloudServiceProvider cloudServiceProvider, IConfiguration configuration)
    {
        _mapper = mapper;
        _identityService = identityService;
        _cloudServiceProvider = cloudServiceProvider;
        _configuration = configuration;
    }
    
    /// <summary>
    /// Creates user with the given credentials 
    /// </summary>
    /// <param name="request"></param>
    /// <param name="cancellationToken"></param>
    /// <returns></returns>
    public async Task<ApiResponse> Handle(CreateUserCommand request, CancellationToken cancellationToken)
    {
        var user = _mapper.Map<ApplicationUser>(request);
        
        user.CreatedOn = DateTime.Now;
        user.UpdatedOn = DateTime.Now;
        
        var result = await _identityService.CreateAsync(user, request.Password);
        
        if(!result.Succeeded)
            return new ApiResponse("An error occurred while creating a user",
                result.Errors.Select(x => x.Description));
        
        await _identityService.AddClaimAsync(user, Claims.User);

        if (result.Succeeded)
        {
            //SendNotificationAsync(user.Id, user.UserName);
            return new ApiResponse("Successfully created a user");
        }

        return new ApiResponse("An error occurred while creating a user",
            result.Errors.Select(x => x.Description));
    }

    private async void SendNotificationAsync(string id, string userName)
    {
        var @event = new UserCreatedEvent(id, userName);
        var sender = _cloudServiceProvider.CreateTopicSender<UserCreatedEvent>(_configuration["AzureServiceBus:Topics:User"]);
        await sender.SendAsync(@event, new MessageMetaData
        {
            CreatedOn = DateTime.Now,
            UpdatedOn = DateTime.Now
        }, CancellationToken.None);
    }
}   