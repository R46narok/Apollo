using Apollo.F1.Compute.Common.Buffers;
using Apollo.F1.Compute.Common.Interfaces;
using Apollo.F1.Compute.Common.LinearAlgebra;
using Apollo.F1.Compute.Cuda.Buffers;
using Apollo.F1.Compute.Cuda.Common;
using Apollo.F1.Compute.Cuda.Common.Execution;
using Apollo.F1.Compute.Cuda.Kernels;
using Apollo.F1.Compute.Cuda.Nvtx;

namespace Apollo.F1.Compute.Cuda.Operations;

public class GpuMatrixOperations : IMatrixHardwareAcceleration
{
    public IRange GetRange () => new NvtxRange();

    
    public MatrixStorage Multiply(MatrixStorage first, MatrixStorage second)
    {
        var output = new MatrixStorage(first.Rows, second.Columns);
        Multiply(first, second, output);
        return output;
    }

    public void Multiply(MatrixStorage first, MatrixStorage second, MatrixStorage output)
    {
        var options = new MultiplicationKernelOptions(first, second, output);
        InvokeKernel<MultiplicationKernel, MultiplicationKernelOptions>(options);
    }

    public MatrixStorage Add(MatrixStorage input, double scalar)
    {
        var output = new MatrixStorage(input.Rows, input.Columns);
        Add(input, output, scalar);
        return output;
    }

    public void Add(MatrixStorage input, MatrixStorage output, double scalar)
    {
        var options = new ScalarKernelOptions(input, output, scalar);
        InvokeKernel<ScalarAdditionKernel, ScalarKernelOptions>(options);
    }

    public double Sum(MatrixStorage matrix)
    {
        var output = new GpuBuffer(new BufferDescriptor {ByteWidth = sizeof(double)});
        var options = new SumKernelOptions(matrix, output);
        InvokeKernel<SumKernel, SumKernelOptions>(options);

        var cpuArray = new double[1];
        GlobalMemory.CopyDeviceToHost(output.Ptr, cpuArray, (int)output.ByteWidth);

        return cpuArray[0];
    }

    public MatrixStorage Multiply(double scalar)
    {
        throw new NotImplementedException();
    }

    public void Multiply(double scalar, MatrixStorage input, MatrixStorage output)
    {
        var options = new ScalarKernelOptions(input, output, scalar);
        InvokeKernel<ScalarMultiplicationKernel, ScalarKernelOptions>(options);
    }

    public MatrixStorage PointwiseLog(MatrixStorage matrix)
    {
        var output = new MatrixStorage(matrix.Rows, matrix.Columns);
        PointwiseLog(matrix, output);
        return output;
    }
    
    public void PointwiseLog(MatrixStorage matrix, MatrixStorage output)
    {
        var options = new PointwiseOperationKernelOptions(matrix, output);
        InvokeKernel<PointwiseLogKernel, PointwiseOperationKernelOptions>(options);
    }
    
    public void ApplySigmoid(MatrixStorage matrix, MatrixStorage output)
    {
        var options = new FunctionKernelOptions(matrix, output);
        InvokeKernel<FunctionSigmoidKernel, FunctionKernelOptions>(options);
    }

    public void ApplySigmoidGradient(MatrixStorage matrix, MatrixStorage output)
    {
        var options = new FunctionKernelOptions(matrix, output);
        InvokeKernel<FunctionSigmoidGradientKernel, FunctionKernelOptions>(options);
    }

    public void InsertColumn(MatrixStorage matrix, MatrixStorage output, double value)
    {
        var options = new InsertKernelOptions(matrix, output, value);
        InvokeKernel<InsertColumnKernel, InsertKernelOptions>(options);
    }

    public MatrixStorage InsertColumn(MatrixStorage matrix, double value)
    {
        var output = new MatrixStorage(matrix.Rows, matrix.Columns + 1);
        InsertColumn(matrix, output, value);
        return output;
    }
    
    public void RemoveColumn(MatrixStorage matrix, MatrixStorage output)
    {
        var options = new RemoveKernelOptions(matrix, output, 0);
        InvokeKernel<RemoveColumnKernel, RemoveKernelOptions>(options);
    }

    public MatrixStorage RemoveColumn(MatrixStorage matrix)
    {
        var output = new MatrixStorage(matrix.Rows, matrix.Columns - 1);
        RemoveColumn(matrix, output);
        return output;
    }
    
    public void InsertRow(MatrixStorage matrix, MatrixStorage output, double value)
    {
        var options = new InsertKernelOptions(matrix, output, value);
        InvokeKernel<InsertRowKernel, InsertKernelOptions>(options);
    }

    public MatrixStorage InsertRow(MatrixStorage matrix, double value)
    {
        var output = new MatrixStorage(matrix.Rows + 1, matrix.Columns);
        InsertRow(matrix, output, value);
        return output;
    }


    public MatrixStorage Add(MatrixStorage first, MatrixStorage second)
    {
        var output = new MatrixStorage(first.Rows, first.Columns);
        Add(first, second, output);
        return output;
    }

    public void Add(MatrixStorage first, MatrixStorage second, MatrixStorage output)
    {
        var options = new PointwiseKernelOptions(first, second, output);
        InvokeKernel<PointwiseAdditionKernel, PointwiseKernelOptions>(options);
    }
    
    public MatrixStorage Subtract(MatrixStorage first, MatrixStorage second)
    {
        var output = new MatrixStorage(first.Rows, first.Columns);
        Subtract(first, second, output);
        return output;
    }
    
    public void Subtract(MatrixStorage first, MatrixStorage second, MatrixStorage output)
    {
        var options = new PointwiseKernelOptions(first, second, output, first.Rows * first.Columns);
        InvokeKernel<PointwiseSubtractionKernel, PointwiseKernelOptions>(options);
    }

    public MatrixStorage Subtract(MatrixStorage first, MatrixStorage second, double scale)
    {
        var output = new MatrixStorage(first.Rows, first.Columns);
        Subtract(first, second, output, scale);
        return output;
    }
    
    public void Subtract(MatrixStorage first, MatrixStorage second, MatrixStorage output, double scale)
    {
        var options = new PointwiseKernelOptions(first, second, output, scale);
        InvokeKernel<PointwiseScaledSubtractionKernel, PointwiseKernelOptions>(options);
    }
    
    public MatrixStorage PointwiseMultiply(MatrixStorage first, MatrixStorage second)
    {
        var output = new MatrixStorage(first.Rows, first.Columns);
        PointwiseMultiply(first, second, output);
        return output;
    }
    
    public void PointwiseMultiply(MatrixStorage first, MatrixStorage second, MatrixStorage output)
    {
       var options = new PointwiseKernelOptions(first, second, output);
       InvokeKernel<PointwiseMultiplicationKernel, PointwiseKernelOptions>(options);
    }

    public MatrixStorage Transpose(MatrixStorage matrix)
    {
        var output = new MatrixStorage(matrix.Columns, matrix.Rows);
        Transpose(matrix, output);
        return output;
    }

    public void Transpose(MatrixStorage matrix, MatrixStorage output)
    {
        var options = new TransposeKernelOptions(matrix, output);
        InvokeKernel<TransposeKernel, TransposeKernelOptions>(options);
    }

    private void InvokeKernel<TKernel, TOptions>(TOptions options)
        where TOptions : KernelOptionsBase
        where TKernel : KernelBase<TOptions>, new()
    {
        var kernel = new TKernel();
        kernel.Invoke(options);
    }
}