FROM mcr.microsoft.com/dotnet/aspnet:6.0 AS base
WORKDIR /app
EXPOSE 80
EXPOSE 443

FROM mcr.microsoft.com/dotnet/sdk:6.0 AS build
WORKDIR /src
COPY ["Services/Apollo.Services.Authorization/Apollo.Services.Authorization.Api/Apollo.Services.Authorization.Api.csproj", "Apollo.Services.Authorization.Api/"]
RUN dotnet restore "Services/Apollo.Services.Authorization/Apollo.Services.Authorization.Api/Apollo.Services.Authorization.Api.csproj"
COPY . .
WORKDIR "/src/Apollo.Services.Authorization.Api"
RUN dotnet build "Apollo.Services.Authorization.Api.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "Apollo.Services.Authorization.Api.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "Apollo.Services.Authorization.Api.dll"]
