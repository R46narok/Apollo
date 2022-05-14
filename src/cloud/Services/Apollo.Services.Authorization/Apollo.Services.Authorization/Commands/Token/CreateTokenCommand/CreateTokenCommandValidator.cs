using Apollo.Services.Authorization.Identity;
using FluentValidation;

namespace Apollo.Services.Authorization.Commands.Token.CreateTokenCommand;

public class CreateTokenCommandValidator : AbstractValidator<CreateTokenCommand>
{
    public CreateTokenCommandValidator(IIdentityService identityService)
    {
        RuleFor(x => new {x.UserName, x.Password})
            .MustAsync(async (pair, _) =>
                await identityService.CheckPasswordAsync(await identityService.FindByNameAsync(pair.UserName), pair.Password))
            .WithErrorCode("401")
            .WithMessage("Wrong credentials");
    }
}