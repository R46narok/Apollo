using System.Diagnostics;
using Apollo.F1.Math.Common.LinearAlgebra;
using Apollo.F1.Math.Cuda;
using Apollo.F1.Math.Cuda.Buffers;
using Apollo.F1.Math.Learning;
using Apollo.F1.Math.Neural;

/*var options = new NeuralNetworkOptions
{
    Layers = new []{ 784, 300, 10}
};
var nn = new NeuralNetwork(options);

int samples = 500;

var x = Matrix<double>.Build.Dense(samples, 784);
var y = Matrix<double>.Build.Dense(samples, 10);

var prediction = Vector<double>.Build.Dense(784);

using var rd = new StreamReader("mnist_train.csv");
int line = -1;
while (!rd.EndOfStream && line <= samples)
{
    line++;
    var splits = rd.ReadLine().Split(',');
    if (line > 0 && line <= samples)
    {
        var dd = Array.ConvertAll(splits, double.Parse);
    
        var label = dd[0];
        y[(line - 1), (int)(label)] = 1;
        for (int i = 0; i < 784; ++i)
            x[line - 1, i] = dd[1 + i];
    }
    
}

Console.WriteLine($"Initial cost: {nn.ComputeCost(x, y)}");
nn.GradientDescent(x, y);
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
        prediction = Vector<double>.Build.Dense(784);
        var label = dd[0];
        for (int i = 0; i < 784; ++i)
            prediction[i] = dd[1 + i] / 256.0;
        prediction = prediction.ToRowMatrix().InsertColumn(0, Vector<double>.Build.Dense(1, 1.0)).Row(0);
        var result = nn.FeedForward(prediction, null);
        var output = Math.Round(result[(int)label, 0]);
        if (output == 0) wrong++;
        else correct++;
    }

}

Console.WriteLine($"Correct: {correct}, Wrong: {wrong}");*/
Matrix.BufferFactory = new GpuBufferFactory();
Matrix.Operations = new GpuMatrixOperations();

var options = new NeuralNetworkOptions
{
    Layers = new []{ 784, 300, 10}
};
var nn = new NeuralNetwork(options);

var x = new Matrix(15, 784);
var cpu = new double[15 * 784];

for (int i = 0; i < 15; ++i)
{
    for (int j = 0; j < 784; ++j)
    {
        var axis = (double)(Random.Shared.Next(0, 2) * 2 - 1);
        var distribution = Random.Shared.NextDouble();
        var value = axis * System.Math.Sqrt(6) * distribution;
        cpu[784 * i + j] = value;
    }
}
x.Buffer.Upload(cpu);

var y = new Matrix(15, 10);
var cpuY = new double[15 * 10];
for (int i = 0; i < 15; ++i)
{
    var idx = Random.Shared.Next(0, 10);
    cpuY[10 * i + idx] = 1.0;
}
y.Buffer.Upload(cpuY);

x = x.InsertColumn(1.0);

var st = Stopwatch.StartNew();
Console.WriteLine(nn.ComputeCost(x, y));
nn.Backpropagate(x, y);

// IHost host = Host.CreateDefaultBuilder(args)
//     .ConfigureServices(services => { services.AddHostedService<Worker>(); })
//     .Build();
//
// await host.RunAsync();