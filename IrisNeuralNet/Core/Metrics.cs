using IrisNeuralNet.MathCore;
using System;


namespace IrisNeuralNet.Core
{
    /// <summary>Метрики качества классификации.</summary>
    public static class Metrics
    {
        public static float Accuracy(ReadOnlySpan<float> predicted, ReadOnlySpan<float> target, int rowWidth)
        {
            SpanGuard.EnsureSameLength(predicted, target);
            SpanGuard.EnsureDivisible(predicted.Length, rowWidth, nameof(rowWidth));

            int correct = 0;
            for (int start = 0; start < predicted.Length; start += rowWidth)
            {
                if (ArgMax(predicted.Slice(start, rowWidth)) == ArgMax(target.Slice(start, rowWidth)))
                {
                    correct++;
                }
            }

            return (float)correct / (predicted.Length / rowWidth);
        }

        public static int ArgMax(ReadOnlySpan<float> row)
        {
            int best = 0;
            for (int i = 1; i < row.Length; i++)
            {
                if (row[i] > row[best])
                {
                    best = i;
                }
            }

            return best;
        }
    }
}
