using Apollo.F1.Compute.Common.Interfaces;
using Apollo.F1.Compute.Common.LinearAlgebra;
using Apollo.F1.Compute.Cuda.Buffers;
using Apollo.F1.Compute.Cuda.Operations;
using Apollo.F1.Compute.Learning;
using Apollo.F1.Compute.Neural;
using Apollo.F1.Compute.Optimization;
using MathNet.Numerics.LinearAlgebra;

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
var cX = Matrix<double>.Build.Dense(samples, 784);
var cY = Matrix<double>.Build.Dense(samples, 10);

while (!rd.EndOfStream && line <= samples)
{
    line++;
    var splits = rd.ReadLine().Split(',');
    if (line > 0 && line <= samples)
    {
        var dd = Array.ConvertAll(splits, double.Parse);
    
        var label = dd[0];
        cpuY[(line - 1) * 10 + (int) label] = 1;
        cY[(line - 1), (int)(label)] = 1;
        for (int i = 0; i < 784; ++i)
        {
            cpuX[(line - 1) * 784 + i] = dd[1 + i] / 256.0;
            cX[line - 1, i] = dd[1 + i] / 256.0;
        }
    }
}

x.Buffer.Upload(cpuX);
y.Buffer.Upload(cpuY);

x = x.InsertColumn(1.0);

cX = cX.InsertColumn(0, Vector<double>.Build.Dense(samples, 1.0));
Matrix<double>[] weights = new Matrix<double>[2];
for (int i = 0; i < weights.Length; i++)
{
    weights[i] = Matrix<double>.Build.Dense(nn._weights[i].Rows, nn._weights[i].Columns);
    var buffer = nn._weights[i].Buffer.Read();
    for (int j = 0; j < nn._weights[i].Rows; ++j)
    for (int k = 0; k < nn._weights[i].Columns; ++k)
        weights[i][j, k] = buffer[j * nn._weights[i].Columns + k];
}

var cpuForwardProp = () =>
{
    var a1 = cX;
    var z2 = a1.Multiply(weights[0].Transpose());

    var a2 = z2.Map(f => 1.0 / (1 + System.Math.Exp(-1 * f)));
    
    a2 = a2.InsertColumn(0,
                Vector<double>.Build.Dense(a2.RowCount, 1.0));
    var z3 = a2.Multiply(weights[1].Transpose());
    var a3 = z3.Map(f => 1.0 / (1 + System.Math.Exp(-1 * f)));
    var h_x = a3;

    return h_x;
};

var gpuForwardProp = () =>
{
    nn.InitFF(x);
    var a1 = x;
    a1.Multiply(nn._weightsTransposed[0], nn._z2);

    nn._z2.ApplySigmoid(nn._z2);
            
    nn._z2.InsertColumn(1.0, nn._a2);
            
    nn._a2.Multiply(nn._weightsTransposed[1], nn._z3);
    nn._z3.ApplySigmoid(nn._z3);

    return nn._z3;
};

var first = cpuForwardProp();
var second = gpuForwardProp().Buffer.Read();
for (int i = 0; i < first.RowCount; ++i)
for (int j = 0; j < first.ColumnCount; j++)
{
    if (Math.Abs(first[i, j] - second[i * first.ColumnCount + j]) > 0.001)
        Console.WriteLine("ERROR");
}
Console.WriteLine($"Initial cost: {nn.ComputeCost(x, y)}");
nn.GradientDescent(x, y);

// IHost host = Host.CreateDefaultBuilder(args)
//     .ConfigureServices(services => { services.AddHostedService<Worker>(); })
//     .Build();
//
// await host.RunAsync();