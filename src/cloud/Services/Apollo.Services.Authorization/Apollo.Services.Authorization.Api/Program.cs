using Apollo.Services.Authorization.Data.Entities;
using Apollo.Services.Authorization.Extensions;
using Apollo.Services.Authorization.Identity;
using FluentValidation;
using MediatR;
using Microsoft.EntityFrameworkCore.ChangeTracking.Internal;
using Rift.CloudProviders.Azure.Common;
using Rift.CloudProviders.Azure.MessageBrokers.Extensions;
using Rift.CloudProviders.Common;
using Rift.Core.Behaviours;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddCors();
builder.Services.AddControllers();

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// Services
builder.AddPersistence();
builder.Services.AddTransient<IIdentityService, IdentityService>();

// Application layer
var assembly = typeof(ApplicationUser).Assembly;
builder.Services.AddMediatR(assembly);
builder.Services.AddTransient(typeof(IPipelineBehavior<,>), typeof(ValidationBehavior<,>));
builder.Services.AddTransient(typeof(IPipelineBehavior<,>), typeof(PerformanceBehavior<,>));
builder.Services.AddValidatorsFromAssembly(assembly);
builder.Services.AddAutoMapper(assembly);

// Cloud
builder.AddAzureServiceBusSenders(assembly);
builder.Services.AddTransient<ICloudServiceProvider, AzureServiceProvider>(provider =>
{
    var resolver = provider.GetService<ServiceBusSenderResolver>()!;
    return new AzureServiceProvider(resolver);
});

// Identity and security
builder.Services.AddIdentity();
builder.AddSecurity();

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.EnsureDatabaseCreated();

app.UseAuthorization();

app.MapControllers();

app.Run();