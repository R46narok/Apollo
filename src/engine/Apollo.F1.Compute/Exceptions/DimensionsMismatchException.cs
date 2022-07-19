using Apollo.F1.Compute.Common.LinearAlgebra;

namespace Apollo.F1.Compute.Exceptions;

public class DimensionsMismatchException : ArgumentException
{
    public Matrix FirstOperand { get; init; }
    public Matrix SecondOperand { get; init; }

    public DimensionsMismatchException(Matrix firstOperand, Matrix secondOperand, string message)
        : base(message)
    {
        FirstOperand = firstOperand;
        SecondOperand = secondOperand;
    }

    public static void ThrowIfNotEqual(Matrix firstOperand, Matrix secondOperand)
    {
        if (firstOperand.Rows != secondOperand.Rows || firstOperand.Columns != secondOperand.Columns)
            throw new DimensionsMismatchException(firstOperand, secondOperand, 
                $"Dimensions not equal [{firstOperand.Rows}x{firstOperand.Columns}] and [{secondOperand.Rows}x{secondOperand.Columns}]");
    }

    public static void ThrowIfNotEqual(Matrix firstOperand, Matrix secondOperand, Matrix output)
    {
        ThrowIfNotEqual(firstOperand, secondOperand);
        ThrowIfNotEqual(firstOperand, output);
    }
}