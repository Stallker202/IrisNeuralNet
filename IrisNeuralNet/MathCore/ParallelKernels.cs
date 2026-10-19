using System;
using System.Threading.Tasks;

namespace IrisNeuralNet.MathCore;

/// <summary>
/// Раздаёт независимые диапазоны строк матричных произведений по ядрам.
/// Span нельзя захватить в лямбду, поэтому память закрепляется (fixed) на время
/// синхронного <see cref="Parallel.For(int, int, Action{int})"/>, а в лямбду уходит адрес.
/// Каждый элемент результата считает ровно один поток тем же кодом, что и последовательная
/// версия, — детерминизм по seed сохраняется.
/// </summary>
internal static unsafe class ParallelKernels
{
    /// <summary>
    /// Порог работы (число умножений), ниже которого накладные расходы пула потоков дороже выигрыша.
    /// 2^22 ≈ 1 мс последовательного SIMD-счёта: Iris/Sonar/Boston остаются однопоточными,
    /// слои MNIST (128 × 784 × 512) распараллеливаются.
    /// </summary>
    public const long WorkThreshold = 1L << 22;

    public static bool ShouldSplit(long work, int independentRows) =>
        work >= WorkThreshold && independentRows > 1 && Environment.ProcessorCount > 1;

    public static void MatMul(
        ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> destination,
        int leftRows, int inner, int rightColumns)
    {
        fixed (float* l = left)
        fixed (float* r = right)
        fixed (float* d = destination)
        {
            nint leftAddress = (nint)l, rightAddress = (nint)r, destinationAddress = (nint)d;
            int rightLength = right.Length;
            ForEachRange(leftRows, (start, count) =>
                VectorOps.MatMulRows(
                    new ReadOnlySpan<float>((float*)leftAddress + (long)start * inner, count * inner),
                    new ReadOnlySpan<float>((float*)rightAddress, rightLength),
                    new Span<float>((float*)destinationAddress + (long)start * rightColumns, count * rightColumns),
                    inner,
                    rightColumns));
        }
    }

    public static void MatMulWithTransposedLeft(
        ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> destination,
        int leftRows, int leftColumns, int rightColumns)
    {
        fixed (float* l = left)
        fixed (float* r = right)
        fixed (float* d = destination)
        {
            nint leftAddress = (nint)l, rightAddress = (nint)r, destinationAddress = (nint)d;
            int leftLength = left.Length, rightLength = right.Length;
            ForEachRange(leftColumns, (start, count) =>
                VectorOps.MatMulWithTransposedLeftRows(
                    new ReadOnlySpan<float>((float*)leftAddress, leftLength),
                    new ReadOnlySpan<float>((float*)rightAddress, rightLength),
                    new Span<float>((float*)destinationAddress + (long)start * rightColumns, count * rightColumns),
                    leftColumns,
                    rightColumns,
                    firstColumn: start));
        }
    }

    public static void MatMulWithTransposedRight(
        ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> destination,
        int leftRows, int inner, int rightRows)
    {
        fixed (float* l = left)
        fixed (float* r = right)
        fixed (float* d = destination)
        {
            nint leftAddress = (nint)l, rightAddress = (nint)r, destinationAddress = (nint)d;
            int rightLength = right.Length;
            ForEachRange(leftRows, (start, count) =>
                VectorOps.MatMulWithTransposedRightRows(
                    new ReadOnlySpan<float>((float*)leftAddress + (long)start * inner, count * inner),
                    new ReadOnlySpan<float>((float*)rightAddress, rightLength),
                    new Span<float>((float*)destinationAddress + (long)start * rightRows, count * rightRows),
                    inner,
                    rightRows));
        }
    }

    /// <summary>Делит [0, total) на непрерывные диапазоны по числу ядер и обрабатывает их параллельно.</summary>
    private static void ForEachRange(int total, Action<int, int> body)
    {
        int partitions = Math.Min(total, Environment.ProcessorCount);
        Parallel.For(0, partitions, partition =>
        {
            int start = (int)((long)total * partition / partitions);
            int end = (int)((long)total * (partition + 1) / partitions);
            body(start, end - start);
        });
    }
}
