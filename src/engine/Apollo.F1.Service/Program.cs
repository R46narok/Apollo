using Apollo.F1.Math.Common.Interfaces;
using Apollo.F1.Math.Common.LinearAlgebra;
using Apollo.F1.Math.Cuda.Buffers;
using Apollo.F1.Math.Cuda.Operations;
using Apollo.F1.Math.Learning;
using Apollo.F1.Math.Neural;
using Apollo.F1.Math.Optimization;

Matrix.BufferFactory = new GlobalMemoryAllocator();
Matrix.Operations = new GpuMatrixOperations();

var options = new NeuralNetworkOptions
{
    Layers = new []{ 784, 300, 10}
};
var nn = new NeuralNetwork(options);

int samples = 700;

var x = new Matrix(samples, 784);
var y = new Matrix(samples, 10);

using var rd = new StreamReader("mnist_train.csv");
int line = -1;

var cpuX = new double[samples * 784];
var cpuY = new double[samples * 10];

while (!rd.EndOfStream && line <= samples)
{
    line++;
    var splits = rd.ReadLine().Split(',');
    if (line > 0 && line <= samples)
    {
        var dd = Array.ConvertAll(splits, double.Parse);
    
        var label = dd[0];
        cpuY[(line - 1) * 10 + (int) label] = 1;
        for (int i = 0; i < 784; ++i)
            cpuX[(line - 1) * 784 + i] = dd[1 + i] / 256.0;
    }
    
}

x.Buffer.Upload(cpuX);
y.Buffer.Upload(cpuY);
x = x.InsertColumn(1.0);

Console.WriteLine($"Initial cost: {nn.ComputeCost(x, y)}");
nn.GradientDescent(x, y);
/*
using var rd2 = new StreamReader("mnist_test.csv");
line = -1;
int correct = 0;
int wrong = 0;
while (!rd.EndOfStream)
{
    line++;
    var splits = rd.ReadLine().Split(',');
    if (line > 0)
    {
        var dd = Array.ConvertAll(splits, double.Parse);
        var prediction = new Matrix(1, 784);
        var cpuData = new double[784];
        var label = dd[0];
        for (int i = 0; i < 784; ++i)
            cpuData[i] = dd[1 + i] / 256.0;
        prediction.Buffer.Upload(cpuData);
        prediction = prediction.InsertColumn(1.0);

        var a1 = prediction;
                        
        var z2 = a1.Multiply(nn._weights[0].Transpose());
        z2.ApplySigmoid(z2);
        var a2 = z2;
        
        a2 = a2.InsertColumn(1.0);
        
        var z3 = a2.Multiply(nn._weights[1].Transpose());
        z3.ApplySigmoid(z3);
        
        Console.WriteLine($"Label {label}: [{string.Join(" ", z3.Buffer.Read())}]");
    }
}*/

// IHost host = Host.CreateDefaultBuilder(args)
//     .ConfigureServices(services => { services.AddHostedService<Worker>(); })
//     .Build();
//
// await host.RunAsync();