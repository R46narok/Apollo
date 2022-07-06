using System.Runtime.InteropServices.ComTypes;
using Apollo.F1.Math.Common.Buffers;
using Apollo.F1.Math.Common.LinearAlgebra;
using Apollo.F1.Math.Cuda.Buffers;
using Apollo.F1.Math.Cuda.Common;
using Apollo.F1.Math.Cuda.Kernels;

namespace Apollo.F1.Math.Cuda;

public class GpuMatrixOperations : IMatrixOperations
{
    public Matrix Multiply(Matrix first, Matrix second)
    {
        var output = new Matrix(first.Rows, second.Columns);
        Multiply(first, second, output);
        return output;
    }

    public void Multiply(Matrix first, Matrix second, Matrix output)
    {
        var options = new MultiplicationKernelOptions(first, second, output);
        InvokeKernel<MultiplicationKernel, MultiplicationKernelOptions>(options);
    }

    public Matrix Add(Matrix input, double scalar)
    {
        var output = new Matrix(input.Rows, input.Columns);
        Add(input, output, scalar);
        return output;
    }

    public void Add(Matrix input, Matrix output, double scalar)
    {
        var options = new ScalarKernelOptions(input, output, scalar);
        InvokeKernel<ScalarAdditionKernel, ScalarKernelOptions>(options);
    }

    public double Sum(Matrix matrix)
    {
        var output = new GpuBuffer(new BufferDescriptor {ByteWidth = sizeof(double)});
        var options = new SumKernelOptions(matrix, output);
        InvokeKernel<SumKernel, SumKernelOptions>(options);

        var cpuArray = new double[1];
        Vram.CopyDeviceToHost(output.Ptr, cpuArray, output.ByteWidth);

        return cpuArray[0];
    }

    public Matrix Multiply(double scalar)
    {
        throw new NotImplementedException();
    }

    public void Multiply(double scalar, Matrix input, Matrix output)
    {
        var options = new ScalarKernelOptions(input, output, scalar);
        InvokeKernel<ScalarMultiplicationKernel, ScalarKernelOptions>(options);
    }

    public Matrix PointwiseLog(Matrix matrix)
    {
        var output = new Matrix(matrix.Rows, matrix.Columns);
        PointwiseLog(matrix, output);
        return output;
    }
    
    public void PointwiseLog(Matrix matrix, Matrix output)
    {
        var options = new PointwiseOperationKernelOptions(matrix, output);
        InvokeKernel<PointwiseLogKernel, PointwiseOperationKernelOptions>(options);
    }
    
    public void ApplySigmoid(Matrix matrix)
    {
        var options = new FunctionKernelOptions(matrix, matrix);
        InvokeKernel<FunctionSigmoidKernel, FunctionKernelOptions>(options);
    }

    public void ApplySigmoidGradient(Matrix matrix)
    {
        var options = new FunctionKernelOptions(matrix, matrix);
        InvokeKernel<FunctionSigmoidGradientKernel, FunctionKernelOptions>(options);
    }

    public void InsertColumn(Matrix matrix, Matrix output, double value)
    {
        var options = new InsertKernelOptions(matrix, output, value);
        InvokeKernel<InsertColumnKernel, InsertKernelOptions>(options);
    }

    public Matrix InsertColumn(Matrix matrix, double value)
    {
        var output = new Matrix(matrix.Rows, matrix.Columns + 1);
        InsertColumn(matrix, output, value);
        return output;
    }
    
    public void RemoveColumn(Matrix matrix, Matrix output)
    {
        var options = new RemoveKernelOptions(matrix, output, 0);
        InvokeKernel<RemoveColumnKernel, RemoveKernelOptions>(options);
    }

    public Matrix RemoveColumn(Matrix matrix)
    {
        var output = new Matrix(matrix.Rows, matrix.Columns - 1);
        RemoveColumn(matrix, output);
        return output;
    }
    
    public void InsertRow(Matrix matrix, Matrix output, double value)
    {
        var options = new InsertKernelOptions(matrix, output, value);
        InvokeKernel<InsertRowKernel, InsertKernelOptions>(options);
    }

    public Matrix InsertRow(Matrix matrix, double value)
    {
        var output = new Matrix(matrix.Rows + 1, matrix.Columns);
        InsertRow(matrix, output, value);
        return output;
    }
    
    public Matrix Add(Matrix first, Matrix second)
    {
        var output = new Matrix(first.Rows, first.Columns);
        Add(first, second, output);
        return output;
    }

    public void Add(Matrix first, Matrix second, Matrix output)
    {
        var options = new PointwiseKernelOptions(first, second, output);
        InvokeKernel<PointwiseAdditionKernel, PointwiseKernelOptions>(options);
    }
    
    public Matrix Subtract(Matrix first, Matrix second)
    {
        var output = new Matrix(first.Rows, first.Columns);
        Subtract(first, second, output);
        return output;
    }
    
    public void Subtract(Matrix first, Matrix second, Matrix output)
    {
        var options = new PointwiseKernelOptions(first, second, output, first.Rows * first.Columns);
        InvokeKernel<PointwiseSubtractionKernel, PointwiseKernelOptions>(options);
    }

    public Matrix Subtract(Matrix first, Matrix second, double scale)
    {
        var output = new Matrix(first.Rows, first.Columns);
        Subtract(first, second, output, scale);
        return output;
    }
    
    public void Subtract(Matrix first, Matrix second, Matrix output, double scale)
    {
        var options = new PointwiseKernelOptions(first, second, output, scale);
        InvokeKernel<PointwiseScaledSubtractionKernel, PointwiseKernelOptions>(options);
    }
    
    public Matrix PointwiseMultiply(Matrix first, Matrix second)
    {
        var output = new Matrix(first.Rows, first.Columns);
        PointwiseMultiply(first, second, output);
        return output;
    }
    
    public void PointwiseMultiply(Matrix first, Matrix second, Matrix output)
    {
       var options = new PointwiseKernelOptions(first, second, output);
       InvokeKernel<PointwiseMultiplicationKernel, PointwiseKernelOptions>(options);
    }

    public Matrix Transpose(Matrix matrix)
    {
        var output = new Matrix(matrix.Columns, matrix.Rows);
        Transpose(matrix, output);
        return output;
    }

    public void Transpose(Matrix matrix, Matrix output)
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