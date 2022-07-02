using Apollo.F1.Math.Common.Buffers;
using Apollo.F1.Math.Cuda.Buffers;
using Apollo.F1.Math.Cuda.Kernels;
using Apollo.F1.Math.Learning;
using Apollo.F1.Math.Neural;
using MathNet.Numerics.LinearAlgebra;

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

using var a = new GpuBuffer(new BufferDescriptor
{
    Usage = BufferUsage.GpuOnly,
    Offset = 0,
    Stride = 0,
    ByteWidth = sizeof(double) * 1024
});

using var b = new GpuBuffer(new BufferDescriptor
{
    Usage = BufferUsage.GpuOnly,
    Offset = 0,
    Stride = 0,
    ByteWidth = sizeof(double) * 1024
});

using var c = new GpuBuffer(new BufferDescriptor
{
    Usage = BufferUsage.GpuOnly,
    Offset = 0,
    Stride = 0,
    ByteWidth = sizeof(double) * 64 * 64
});
var first = new double[1024];
var second = new double[1024];
for (int i = 0; i < 64; ++i)
{
    for (int j = 0; j < 16; j++)
    {
        first[16 * i + j] = Random.Shared.Next(1, 10);
    }
}

for (int i = 0; i < 16; ++i)
{
    for (int j = 0; j < 64; j++)
    {
        second[64 * i + j] = Random.Shared.Next(2, 10);
    }
}

Vram.CopyHostToDevice(first, a.Ptr, first.Length * sizeof(double));
Vram.CopyHostToDevice(second, b.Ptr, second.Length * sizeof(double));

var kernel = new MultiplicationKernel(64, 16, 64);
kernel.Invoke(new []{a, b, c});

var third = new double[64 *64];
Vram.CopyDeviceToHost(c.Ptr, third, 64 *64  * sizeof(double));


Console.WriteLine("Transposed:");

for (int i = 0; i < 64; ++i)
{
    for (int j = 0; j < 64; ++j)
        Console.Write(third[64 * i + j] + " ");
    Console.WriteLine();
}
// IHost host = Host.CreateDefaultBuilder(args)
//     .ConfigureServices(services => { services.AddHostedService<Worker>(); })
//     .Build();
//
// await host.RunAsync();