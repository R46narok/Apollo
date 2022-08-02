using Apollo.F1.Compute.Common.LinearAlgebra;
using Apollo.F1.Compute.Cuda.Buffers;
using Apollo.F1.Compute.Cuda.Common.Interop;
using Apollo.F1.Compute.Cuda.Operations;
using Apollo.F1.Compute.Learning.Neural;
using Apollo.F1.Compute.Optimization.Algorithms;
using Apollo.F1.Platform.Windows.FileSystem;

MatrixStorage.BufferFactory = new GlobalMemoryAllocator();
MatrixStorage.Operations = new GpuMatrixOperations();

var options = new NeuralNetworkOptions
{
    Layers = new []{ 784, 300, 10},
    Distribution = new UniformDistribution(Math.Sqrt(6))
};
var nn = new NeuralNetwork(options);

int samples = 5;
var x = new MatrixStorage(samples, 784);
var y = new MatrixStorage(samples, 10);

var assert = new CudaAssert();

var procedure = new GradientDescent<NeuralOptimizationContext, NeuralPredictionContext>(0.25, 4000);
procedure.Optimize(nn, x, y);

var predictionContext = new NeuralPredictionContext();
predictionContext.AllocateMemoryForPredictionBatch(nn.Parameters, 1);

// IHost host = Host.CreateDefaultBuilder(args)
//     .ConfigureServices(services => { services.AddPlatform(new[] {typeof(WindowsPlatform).Assembly}); })
//     .Build();
//  await host.RunAsync();