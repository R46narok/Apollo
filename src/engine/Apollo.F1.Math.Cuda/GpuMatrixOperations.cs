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
        var kernel = new MultiplicationKernel(first.Rows, first.Columns, second.Columns);
        var buffers = ConvertToGpuBuffers(new[] {first.Buffer, second.Buffer, output.Buffer});
        kernel.Invoke(buffers);
    }

    public Matrix Multiply(double scalar)
    {
        throw new NotImplementedException();
    }

    public void Multiply(double scalar, Matrix input, Matrix output)
    {
        var kernel = new ScalarMultiplicationKernel(scalar);
        var buffers = ConvertToGpuBuffers(new[] {input.Buffer, output.Buffer});
        kernel.Invoke(buffers);
    }

    public void PointwiseLog(Matrix matrix)
    {
        var kernel = new PointwiseLogKernel();
        var buffers = ConvertToGpuBuffers(new[] {matrix.Buffer, matrix.Buffer});
        kernel.Invoke(buffers);
    }
    
    public void ApplySigmoid(Matrix matrix)
    {
        var kernel = new FunctionSigmoidKernel();
        var buffers = ConvertToGpuBuffers(new[] {matrix.Buffer});
        kernel.Invoke(buffers);
    }

    public void ApplySigmoidGradient(Matrix matrix)
    {
        var kernel = new FunctionSigmoidGradientKernel();
        var buffers = ConvertToGpuBuffers(new[] {matrix.Buffer});
        kernel.Invoke(buffers);
    }

    public void InsertColumn(Matrix matrix, Matrix output, double value)
    {
        var kernel = new InsertColumnKernel(matrix.Rows, matrix.Columns, value);
        var buffers = ConvertToGpuBuffers(new[] {matrix.Buffer, output.Buffer});
        kernel.Invoke(buffers);
    }

    public Matrix InsertColumn(Matrix matrix, double value)
    {
        var output = new Matrix(matrix.Rows, matrix.Columns + 1);
        InsertColumn(matrix, output, value);
        return output;
    }
    
    
    public void RemoveColumn(Matrix matrix, Matrix output)
    {
        var kernel = new RemoveColumnKernel(matrix.Rows, matrix.Columns);
        var buffers = ConvertToGpuBuffers(new[] {matrix.Buffer, output.Buffer});
        kernel.Invoke(buffers);
    }

    public Matrix RemoveColumn(Matrix matrix)
    {
        var output = new Matrix(matrix.Rows, matrix.Columns - 1);
        RemoveColumn(matrix, output); 
        return output;
    }
    
    public void InsertRow(Matrix matrix, Matrix output, double value)
    {
        var kernel = new InsertRowKernel(matrix.Rows, matrix.Columns, value);
        var buffers = ConvertToGpuBuffers(new[] {matrix.Buffer, output.Buffer});
        kernel.Invoke(buffers);
    }

    public Matrix InsertRow(Matrix matrix, double value)
    {
        var output = new Matrix(matrix.Rows + 1, matrix.Columns);
        InsertRow(matrix, output, value);
        return output;
    }
    public Matrix Add(Matrix first, Matrix second)
    {
        EnsureEqualDimensions(first, second);
        
        var output = new Matrix(first.Rows, first.Columns);
        Add(first, second, output);
        
        return output;
    }

    public void Add(Matrix first, Matrix second, Matrix output)
    {
        EnsureEqualDimensions(first, second);
        
        var kernel = new PointwiseAdditionKernel();
        var buffers = ConvertToGpuBuffers(new[] {first.Buffer, second.Buffer, output.Buffer});
        kernel.Invoke(buffers);
    }
    
    public Matrix Subtract(Matrix first, Matrix second)
    {
        EnsureEqualDimensions(first, second);
        
        var output = new Matrix(first.Rows, first.Columns);
        Subtract(first, second, output);
        
        return output;
    }
    
    public void Subtract(Matrix first, Matrix second, Matrix output)
    {
        EnsureEqualDimensions(first, second);
        
        var kernel = new PointwiseSubtractionKernel();
        var buffers = ConvertToGpuBuffers(new[] {first.Buffer, second.Buffer, output.Buffer});
        kernel.Invoke(buffers);
    }
    
    public Matrix PointwiseMultiply(Matrix first, Matrix second)
    {
        EnsureEqualDimensions(first, second);
        
        var output = new Matrix(first.Rows, first.Columns);
        PointwiseMultiply(first, second, output);
        
        return output;
    }
    
    public void PointwiseMultiply(Matrix first, Matrix second, Matrix output)
    {
        EnsureEqualDimensions(first, second);
        
        var kernel = new PointwiseMultiplicationKernel();
        var buffers = ConvertToGpuBuffers(new[] {first.Buffer, second.Buffer, output.Buffer});
        kernel.Invoke(buffers);
    }

    public Matrix Transpose(Matrix matrix)
    {
        var output = new Matrix(matrix.Columns, matrix.Rows);
        Transpose(matrix, output);

        return output;
    }

    public void Transpose(Matrix matrix, Matrix output)
    {
        var kernel = new TransposeKernel(matrix.Rows, matrix.Columns);
        var buffers = ConvertToGpuBuffers(new[] { matrix.Buffer, output.Buffer});
        kernel.Invoke(buffers);
    }

    private void EnsureEqualDimensions(Matrix first, Matrix second)
    {
        if (!(first.Rows == second.Rows && first.Columns == second.Columns))
        {
            throw new ArgumentException();
        }
    }

    private GpuBuffer[] ConvertToGpuBuffers(IBuffer[] buffers)
    {
        return Array.ConvertAll(buffers, b => (GpuBuffer) b);
    }
}