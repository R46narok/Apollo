using System.Buffers;
using System.Diagnostics;
using System.Text;
using Apollo.F1.Math.Acceleration;
using Apollo.F1.Math.Exceptions;

namespace Apollo.F1.Math;

[DebuggerDisplay("Dimensions[{Rows}x{Columns}]")]
public class Matrix
{
    private readonly double[] _elements;
    public uint Rows { get; private set; }
    public uint Columns { get; private set; }

    public Span<double> AsRowVector(uint row)
    {
        return _elements.AsSpan().Slice((int) (Columns * row), (int) Columns);
    }

    public double this[long row, long column]
    {
        get => _elements[Columns * row + column];
        set => _elements[Columns * row + column] = value;
    }
    
    public Matrix(uint rows, uint columns)
    {
        Rows = rows;
        Columns = columns;
        _elements = new double[rows * columns];
    }

    public Matrix(double[] elements, uint rows, uint columns)
    {
        _elements = elements;
        Rows = rows;
        Columns = columns;
    }
    
    public Matrix Add(Matrix other)
    {
        EnsureEqualDimensions(this, other);
        var first = _elements.AsSpan();
        var second = other._elements.AsSpan();

        var matrix = new Matrix(Rows, Columns);
        CpuMatrixOperations.Add(matrix._elements.AsSpan(), first, second);

        return matrix;
    }
    
    public Matrix Subtract(Matrix other)
    {
        EnsureEqualDimensions(this, other);
        var first = _elements.AsSpan();
        var second = other._elements.AsSpan();
    
        var matrix = new Matrix(Rows, Columns);
        CpuMatrixOperations.Subtract(matrix._elements.AsSpan(), first, second);
    
        return matrix;
    }
        
    public Matrix Dot(Matrix other)
    {
        if (Columns != other.Rows)
            throw new DimensionsException();

        var matrix = new Matrix(Rows, other.Columns);

        for (int i = 0; i < Rows; ++i)
        {
            for (int j = 0; j < other.Columns; ++j)
            {
                double sum = 0.0;
                for (int k = 0; k < other.Rows; k++)
                    sum += this[i, k] * other[k, j];

                matrix[i, j] = sum;
            }
        }
        
        return matrix;
    }

    public void Apply(Func<double, double> function)
    {
        for (int i = 0; i < _elements.Length; ++i)
            _elements[i] = function(_elements[i]);
    }
    
    private static void EnsureEqualDimensions(Matrix first, Matrix second)
    {
        if (!(first.Rows == second.Rows && first.Columns == second.Columns))
            throw new DimensionsException();
    }

    public void Randomize(int n)
    {
        var span = _elements.AsSpan();
        CpuMatrixOperations.Randomize(span, n);
    }

    public Matrix Flatten(VectorType type)
    {
        Matrix result;
        switch (type)
        {
            case VectorType.Row:
                result = new Matrix(1, Rows * Columns);
                break;
            case VectorType.Column:
                result = new Matrix(Rows * Columns, 1);
                break;
            default:
                result = null!;
                break;
        }

        for (int i = 0; i < Rows; ++i)
        {
            for (int j = 0; j < Columns; ++j)
            {
                if (type == VectorType.Column)
                    result[i * Columns + j, 0] = this[i, j];
                else
                    result[0, i * Columns + j] = this[i, j];
            }
        }
        
        return result;
    }
    
    public override string ToString()
    {
        var builder = new StringBuilder();

        builder.Append($"Dimensions [{Rows}x{Columns}]");

        for (int i = 0; i < Rows; ++i)
        {
            builder.AppendLine();
            builder.Append("|");
            for (int j = 0; j < Columns; ++j)
            {
                if (j > 0) builder.Append(" ");
                builder.Append(this[i, j]);
                if (j == Columns - 1) builder.Append("|");
            }
        }
        
        return builder.ToString();
    }
}