using System;
using System.Numerics;

namespace IrisNeuralNet.MathCore;

/// <summary>
/// SIMD-примитивы над непрерывной float-памятью (row-major).
/// Все методы пишут результат в буфер вызывающего: ноль аллокаций в горячем пути.
/// Большие матричные произведения (от <see cref="ParallelKernels.WorkThreshold"/> умножений)
/// раздаются по ядрам; каждая строка результата считается одним потоком в том же порядке,
/// что и последовательно, поэтому результат бит-в-бит совпадает.
/// </summary>
public static class VectorOps
{
    public static void Add(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> destination)
    {
        SpanGuard.EnsureSameLength(left, right, destination);

        int i = 0;
        for (; i <= left.Length - Vector<float>.Count; i += Vector<float>.Count)
        {
            var sum = new Vector<float>(left.Slice(i, Vector<float>.Count))
                    + new Vector<float>(right.Slice(i, Vector<float>.Count));
            sum.CopyTo(destination.Slice(i));
        }

        for (; i < left.Length; i++) destination[i] = left[i] + right[i];
    }

    public static void Scale(ReadOnlySpan<float> source, float factor, Span<float> destination)
    {
        SpanGuard.EnsureSameLength(source, destination);
        var broadcast = new Vector<float>(factor);

        int i = 0;
        for (; i <= source.Length - Vector<float>.Count; i += Vector<float>.Count)
        {
            (new Vector<float>(source.Slice(i, Vector<float>.Count)) * broadcast)
                .CopyTo(destination.Slice(i));
        }

        for (; i < source.Length; i++) destination[i] = source[i] * factor;
    }

    /// <summary>destination[i] += source[i] * factor. Поэлементно, поэтому source и destination могут совпадать.</summary>
    public static void MultiplyAdd(ReadOnlySpan<float> source, float factor, Span<float> destination)
    {
        SpanGuard.EnsureSameLength(source, destination);
        var broadcast = new Vector<float>(factor);

        int i = 0;
        for (; i <= source.Length - Vector<float>.Count; i += Vector<float>.Count)
        {
            var fused = new Vector<float>(source.Slice(i, Vector<float>.Count)) * broadcast
                      + new Vector<float>(destination.Slice(i, Vector<float>.Count));
            fused.CopyTo(destination.Slice(i));
        }

        for (; i < source.Length; i++) destination[i] += source[i] * factor;
    }

    public static float Max(ReadOnlySpan<float> source)
    {
        float max = source[0];
        for (int i = 1; i < source.Length; i++) max = MathF.Max(max, source[i]);
        return max;
    }

    public static float Sum(ReadOnlySpan<float> source)
    {
        var accumulator = Vector<float>.Zero;

        int i = 0;
        for (; i <= source.Length - Vector<float>.Count; i += Vector<float>.Count)
        {
            accumulator += new Vector<float>(source.Slice(i, Vector<float>.Count));
        }

        float sum = Vector.Sum(accumulator);
        for (; i < source.Length; i++) sum += source[i];
        return sum;
    }

    public static float Dot(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
    {
        SpanGuard.EnsureSameLength(left, right);

        var accumulator = Vector<float>.Zero;
        int i = 0;
        for (; i <= left.Length - Vector<float>.Count; i += Vector<float>.Count)
        {
            accumulator += new Vector<float>(left.Slice(i, Vector<float>.Count))
                         * new Vector<float>(right.Slice(i, Vector<float>.Count));
        }

        float dot = Vector.Sum(accumulator);
        for (; i < left.Length; i++) dot += left[i] * right[i];
        return dot;
    }

    /// <summary>
    /// Прибавляет вектор <paramref name="row"/> к каждой строке матрицы на месте.
    /// </summary>
    public static void AddRowBroadcast(Span<float> matrix, ReadOnlySpan<float> row, int rowWidth)
    {
        SpanGuard.EnsureLength(row, rowWidth, nameof(row));
        SpanGuard.EnsureDivisible(matrix.Length, rowWidth, nameof(matrix));

        for (int start = 0; start < matrix.Length; start += rowWidth)
        {
            Span<float> target = matrix.Slice(start, rowWidth);
            Add(target, row, target);
        }
    }

    /// <summary>
    /// left [leftRows x inner] * right [inner x rightColumns] → destination [leftRows x rightColumns].
    /// Порядок циклов i→k→j: внутренняя операция — векторный MultiplyAdd по строке right.
    /// </summary>
    public static void MatMul(
        ReadOnlySpan<float> left,
        ReadOnlySpan<float> right,
        Span<float> destination,
        int leftRows,
        int inner,
        int rightColumns)
    {
        SpanGuard.EnsureLength(left, leftRows * inner, nameof(left));
        SpanGuard.EnsureLength(right, inner * rightColumns, nameof(right));
        SpanGuard.EnsureLength(destination, leftRows * rightColumns, nameof(destination));

        if (ParallelKernels.ShouldSplit((long)leftRows * inner * rightColumns, leftRows))
        {
            ParallelKernels.MatMul(left, right, destination, leftRows, inner, rightColumns);
            return;
        }

        MatMulRows(left, right, destination, inner, rightColumns);
    }

    /// <summary>
    /// destination = left^T * right: left [leftRows x leftColumns], right [leftRows x rightColumns],
    /// destination [leftColumns x rightColumns]. Используется для dW = Xᵀ·δ.
    /// </summary>
    public static void MatMulWithTransposedLeft(
        ReadOnlySpan<float> left,
        ReadOnlySpan<float> right,
        Span<float> destination,
        int leftRows,
        int leftColumns,
        int rightColumns)
    {
        SpanGuard.EnsureLength(left, leftRows * leftColumns, nameof(left));
        SpanGuard.EnsureLength(right, leftRows * rightColumns, nameof(right));
        SpanGuard.EnsureLength(destination, leftColumns * rightColumns, nameof(destination));

        if (ParallelKernels.ShouldSplit((long)leftRows * leftColumns * rightColumns, leftColumns))
        {
            ParallelKernels.MatMulWithTransposedLeft(left, right, destination, leftRows, leftColumns, rightColumns);
            return;
        }

        MatMulWithTransposedLeftRows(left, right, destination, leftColumns, rightColumns, firstColumn: 0);
    }

    /// <summary>
    /// destination = left * right^T: left [leftRows x inner], right [rightRows x inner],
    /// destination [leftRows x rightRows]. Используется для dX = δ·Wᵀ (без копии транспонированных весов).
    /// </summary>
    public static void MatMulWithTransposedRight(
        ReadOnlySpan<float> left,
        ReadOnlySpan<float> right,
        Span<float> destination,
        int leftRows,
        int inner,
        int rightRows)
    {
        SpanGuard.EnsureLength(left, leftRows * inner, nameof(left));
        SpanGuard.EnsureLength(right, rightRows * inner, nameof(right));
        SpanGuard.EnsureLength(destination, leftRows * rightRows, nameof(destination));

        if (ParallelKernels.ShouldSplit((long)leftRows * inner * rightRows, leftRows))
        {
            ParallelKernels.MatMulWithTransposedRight(left, right, destination, leftRows, inner, rightRows);
            return;
        }

        MatMulWithTransposedRightRows(left, right, destination, inner, rightRows);
    }

    /// <summary>Последовательное ядро MatMul для диапазона строк left/destination.</summary>
    internal static void MatMulRows(
        ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> destination, int inner, int rightColumns)
    {
        int rows = left.Length / inner;
        for (int row = 0; row < rows; row++)
        {
            Span<float> destinationRow = destination.Slice(row * rightColumns, rightColumns);
            destinationRow.Clear();

            ReadOnlySpan<float> leftRow = left.Slice(row * inner, inner);
            for (int k = 0; k < inner; k++)
            {
                float factor = leftRow[k];
                if (factor != 0f)
                {
                    MultiplyAdd(right.Slice(k * rightColumns, rightColumns), factor, destinationRow);
                }
            }
        }
    }

    /// <summary>
    /// Последовательное ядро leftᵀ·right для строк результата
    /// [firstColumn, firstColumn + destinationRows.Length / rightColumns).
    /// </summary>
    internal static void MatMulWithTransposedLeftRows(
        ReadOnlySpan<float> left,
        ReadOnlySpan<float> right,
        Span<float> destinationRows,
        int leftColumns,
        int rightColumns,
        int firstColumn)
    {
        int leftRows = left.Length / leftColumns;
        int columnCount = destinationRows.Length / rightColumns;
        for (int c = 0; c < columnCount; c++)
        {
            int k = firstColumn + c;
            Span<float> destinationRow = destinationRows.Slice(c * rightColumns, rightColumns);
            destinationRow.Clear();

            for (int i = 0; i < leftRows; i++)
            {
                float factor = left[i * leftColumns + k];
                if (factor != 0f)
                {
                    MultiplyAdd(right.Slice(i * rightColumns, rightColumns), factor, destinationRow);
                }
            }
        }
    }

    /// <summary>Последовательное ядро left·rightᵀ для диапазона строк left/destination.</summary>
    internal static void MatMulWithTransposedRightRows(
        ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> destination, int inner, int rightRows)
    {
        int rows = left.Length / inner;
        for (int row = 0; row < rows; row++)
        {
            ReadOnlySpan<float> leftRow = left.Slice(row * inner, inner);
            Span<float> destinationRow = destination.Slice(row * rightRows, rightRows);
            for (int k = 0; k < rightRows; k++)
            {
                destinationRow[k] = Dot(leftRow, right.Slice(k * inner, inner));
            }
        }
    }
}
