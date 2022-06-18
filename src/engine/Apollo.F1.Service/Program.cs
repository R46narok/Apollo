using Apollo.F1.Math;
using Apollo.F1.Math.Neural;
using Apollo.F1.Service;

var architecture = new uint[] { 3, 3, 2};
var nn = new NeuralNetwork(architecture);

var x = new Matrix(new[]
{
    1.0, 10.0, 20.0, 30
}, 1, 4);

var y = new Matrix(new[]
{
    1.0, 0.5
}, 1, 4);
Console.WriteLine(nn.ComputeCost(x, y));

IHost host = Host.CreateDefaultBuilder(args)
    .ConfigureServices(services => { services.AddHostedService<Worker>(); })
    .Build();

await host.RunAsync();