using Apollo.Services.Authorization.Commands.Token.CreateTokenCommand;
using Apollo.Services.Authorization.DTOs;
using AutoMapper;
using MediatR;
using Microsoft.AspNetCore.Mvc;

namespace Apollo.Services.Authorization.Api.Controllers;


[ApiController]
[Route("/api/[controller]")]
public class TokenController : ControllerBase
{
    private readonly IMapper _mapper;
    private readonly IMediator _mediator;

    public TokenController(IMapper mapper, IMediator mediator)
    {
        _mapper = mapper;
        _mediator = mediator;
    }
    
    [HttpPost]
    [Consumes("application/json")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status401Unauthorized)]
    public async Task<IActionResult> GetToken([FromBody] UserCredentialsDto credentialsDto)
    {
        var command = _mapper.Map<CreateTokenCommand>(credentialsDto);
        var response = await _mediator.Send(command);

        if (response.IsValid)
            return Ok(response);
        
        return Unauthorized(response);
    }
}