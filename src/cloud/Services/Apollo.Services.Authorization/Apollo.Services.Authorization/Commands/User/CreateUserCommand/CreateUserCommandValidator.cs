using Apollo.Services.Authorization.Identity;
using FluentValidation;

namespace Apollo.Services.Authorization.Commands.User.CreateUserCommand;

public class CreateUserCommandValidator : AbstractValidator<CreateUserCommand>
{
    public CreateUserCommandValidator(IIdentityService identityService)
    {
        RuleFor(command => command.UserName)
            .MustAsync(async (userName, _) =>
                await identityService.FindByNameAsync(userName) is null)
            .WithErrorCode("400")
            .WithMessage("User already exists");
    }
}