using Apollo.F1.Math.Common.Buffers;
using Apollo.F1.Math.Common.LinearAlgebra;
using Apollo.F1.Math.Cuda.Buffers;
using Apollo.F1.Math.Cuda.Common;
using Apollo.F1.Math.Cuda.Kernels;

namespace Apollo.F1.Math.Cuda;

public class GpuMatrixOperations : IMatrixOperations
{
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