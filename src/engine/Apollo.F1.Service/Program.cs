using Apollo.F1.Math.Neural;
using MathNet.Numerics.LinearAlgebra;

var nn = new NeuralNetwork(new[] {3, 3, 2});
var x = Matrix<double>.Build.Dense(1, 4);

x[0, 0] = 1.0;
x[0, 1] = 10;
x[0, 2] = 20;
x[0, 3] = 30;

var y = Matrix<double>.Build.Dense(1, 2);
y[0, 0] = 1.0;
y[0, 1] = 0.5;

nn.Backpropagation(x, y);
// IHost host = Host.CreateDefaultBuilder(args)
//     .ConfigureServices(services => { services.AddHostedService<Worker>(); })
//     .Build();
//
// await host.RunAsync();