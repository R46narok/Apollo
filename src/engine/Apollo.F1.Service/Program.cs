using Apollo.F1.Compute.Common.Buffers;
using Apollo.F1.Compute.Common.LinearAlgebra;
using Apollo.F1.Compute.Cuda.Buffers;
using Apollo.F1.Compute.Cuda.Operations;
using Apollo.F1.Compute.Learning;
using Apollo.F1.Compute.Neural;
using Apollo.F1.Compute.Optimization;

MatrixStorage.BufferFactory = new GlobalMemoryAllocator();
MatrixStorage.Operations = new GpuMatrixOperations();

var options = new NeuralNetworkOptions
{
    Layers = new []{ 784, 300, 10}
};
var nn = new NeuralNetwork(options);

int samples = 700;

var x = new MatrixStorage(samples, 784);
var y = new MatrixStorage(samples, 10);

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
        {
            cpuX[(line - 1) * 784 + i] = dd[1 + i] / 256.0;
        }
    }
}

x.Buffer.Upload(cpuX);
y.Buffer.Upload(cpuY);

x = x.InsertColumn(1.0);
nn.InitFF(x);
nn.GradientDescent(x, y);

using var rd2 = new StreamReader("mnist_test.csv");
line = -1;
int correct = 0;
int wrong = 0;

while (!rd.EndOfStream)
{
    line++;
    var splits = rd.ReadLine()!.Split(',');
    if (line > 0)
    {
        var dd = Array.ConvertAll(splits, double.Parse);
        var prediction = new MatrixStorage(1, 784);
        var cpuData = new double[784];
        var label = dd[0];
        for (int i = 0; i < 784; ++i)
            cpuData[i] = dd[1 + i] / 256.0;
        prediction.Buffer.Upload(cpuData);
        prediction = prediction.InsertColumn(1.0);
        if (line == 1) nn.InitFF(prediction);

        var result = nn.FeedForward(prediction);
        var buffer = result.Buffer.Read();
        var output = Math.Round(buffer[(int)label]);
        
        if (output == 0) wrong++;
        else correct++;
    }
}

Console.WriteLine($"Right: {correct}, wrong {wrong}");
// IHost host = Host.CreateDefaultBuilder(args)
//     .ConfigureServices(services => { services.AddHostedService<Worker>(); })
//     .Build();
//
// await host.RunAsync();