using Apollo.F1.Compute.Common.Buffers;
using Apollo.F1.Compute.Common.LinearAlgebra;
using Apollo.F1.Compute.Cuda.Buffers;
using Apollo.F1.Compute.Cuda.Common.Interop;
using Apollo.F1.Compute.Cuda.Operations;
using Apollo.F1.Compute.Learning;
using Apollo.F1.Compute.Learning.Neural;
using Apollo.F1.Compute.Optimization;
using Apollo.F1.Compute.Optimization.Algorithms;

MatrixStorage.BufferFactory = new GlobalMemoryAllocator();
MatrixStorage.Operations = new GpuMatrixOperations();

var sqrt = Math.Sqrt(6);
var options = new NeuralNetworkOptions
{
    Layers = new []{ 784, 300, 10},
    Distribution = new UniformDistribution(sqrt)
};
var nn = new NeuralNetwork(options);

int samples = 40000;

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

var assert = new CudaAssert();

x = x.InsertColumn(1.0);
var procedure = new GradientDescent<NeuralOptimizationContext, NeuralPredictionContext>(0.25, 4000);
procedure.Optimize(nn, x, y);

using var rd2 = new StreamReader("mnist_test.csv");
line = -1;
int correct = 0;
int wrong = 0;

var predictionContext = new NeuralPredictionContext();
predictionContext.AllocateMemoryForPredictionBatch(nn.Parameters, 1);
while (!rd.EndOfStream)
{
    line++;
    var splits = rd.ReadLine().Split(',');
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


        var result = nn.FeedForward(prediction, predictionContext);
        var buffer = result.Buffer.Read();
        var output = Math.Round(buffer[(int)label]);
        
        if (output == 0) wrong++;
        else correct++;
        // Console.WriteLine($"Label {label}: [{string.Join(" ", prediction.Buffer.Read())}]");
        
    }
}


Console.WriteLine($"Right: {correct}, wrong {wrong}");
// IHost host = Host.CreateDefaultBuilder(args)
//     .ConfigureServices(services => { services.AddHostedService<Worker>(); })
//     .Build();
//
// await host.RunAsync();