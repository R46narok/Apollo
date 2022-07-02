using Apollo.F1.Math.Learning;
using Apollo.F1.Math.Neural;
using MathNet.Numerics.LinearAlgebra;

var options = new NeuralNetworkOptions
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

Console.WriteLine($"Correct: {correct}, Wrong: {wrong}");

// IHost host = Host.CreateDefaultBuilder(args)
//     .ConfigureServices(services => { services.AddHostedService<Worker>(); })
//     .Build();
//
// await host.RunAsync();