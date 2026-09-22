using IrisNeuralNet.MathCore;
using System;


namespace IrisNeuralNet.Core
{
    /// <summary>Категориальная кросс-энтропия; градиент усреднён по батчу.</summary>
    public sealed class CategoricalCrossEntropyLoss : ILossFunction
    {
        private const float ProbabilityFloor = 1e-38f;

        public float Compute(ReadOnlySpan<float> predicted, ReadOnlySpan<float> target, int rowWidth)
        {
            EnsureBatch(predicted, target, rowWidth);

            float total = 0f;
            for (int start = 0; start < predicted.Length; start += rowWidth)
            {
                ReadOnlySpan<float> row = predicted.Slice(start, rowWidth);
                ReadOnlySpan<float> targetRow = target.Slice(start, rowWidth);
                for (int i = 0; i < rowWidth; i++)
                {
                    total -= targetRow[i] * MathF.Log(MathF.Max(row[i], ProbabilityFloor));
                }
            }

            return total / (predicted.Length / rowWidth);
        }

        public void Backward(ReadOnlySpan<float> predicted, ReadOnlySpan<float> target, int rowWidth, Span<float> destination)
        {
            EnsureBatch(predicted, target, rowWidth);
            SpanGuard.EnsureSameLength(predicted, destination);

            float rowCount = predicted.Length / rowWidth;
            for (int i = 0; i < predicted.Length; i++)
            {
                destination[i] = -target[i] / MathF.Max(predicted[i], ProbabilityFloor) / rowCount;
            }
        }

        private static void EnsureBatch(ReadOnlySpan<float> predicted, ReadOnlySpan<float> target, int rowWidth)
        {
            SpanGuard.EnsureSameLength(predicted, target);
            SpanGuard.EnsureDivisible(predicted.Length, rowWidth, nameof(rowWidth));
        }
    }
}
