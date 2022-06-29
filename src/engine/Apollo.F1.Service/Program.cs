using System.Threading.Channels;
using Apollo.F1.Math.Neural;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Optimization;
using MathNet.Numerics.Optimization.ObjectiveFunctions;

var nn = new NeuralNetwork(new[] {784, 300, 10});

var x = Matrix<double>.Build.Dense(1000, 784);
var y = Matrix<double>.Build.Dense(1000, 10);

var prediction = Vector<double>.Build.Dense(784);
var l = 0.0;

using var rd = new StreamReader("mnist_train.csv");
int line = -1;
while (!rd.EndOfStream)
{
    line++;
    var splits = rd.ReadLine().Split(',');
    if (line > 0 && line <= 1000)
    {
        var dd = Array.ConvertAll(splits, double.Parse);
    
        var label = dd[0];
        y[(line - 1), (int)(label)] = 1;
        for (int i = 0; i < 784; ++i)
            x[line - 1, i] = dd[1 + i];
    }
    else if (line >= 1200)
    {
        var dd = Array.ConvertAll(splits, double.Parse);
    
        var label = dd[0];
        l = label;
        for (int i = 0; i < 784; ++i)
            prediction[i] = dd[1 + i] / 256.0;
    }
    
}

Console.WriteLine(nn.ComputeCost(x, y));
nn.Backpropagation(x, y);
nn.GradientDescent(x, y);

Console.WriteLine(l);
prediction = prediction.ToRowMatrix().InsertColumn(0, Vector<double>.Build.Dense(1, 1.0)).Row(0);
Console.WriteLine(nn.ForwardPropagation(prediction, null));

// IHost host = Host.CreateDefaultBuilder(args)
//     .ConfigureServices(services => { services.AddHostedService<Worker>(); })
//     .Build();
//
// await host.RunAsync();