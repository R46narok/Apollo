using Apollo.Services.Authorization.Commands.User.CreateUserCommand;
using AutoMapper;
using MediatR;
using Microsoft.AspNetCore.Mvc;

namespace Apollo.Services.Authorization.Api.Controllers;

[ApiController]
[Route("/api/[controller]")]
public class UserController : ControllerBase
{
    private IMapper _mapper;
    private readonly IMediator _mediator;

    public UserController(IMediator mediator, IMapper mapper)
    {
        _mapper = mapper;
        _mediator = mediator;
    }
    
    [HttpPost]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    public async Task<IActionResult> RegisterAsync([FromBody] CreateUserCommand command)
    {
        var response = await _mediator.Send(command);

        if (response.IsValid)
            return Ok(response);
        return BadRequest(response);
    }
}