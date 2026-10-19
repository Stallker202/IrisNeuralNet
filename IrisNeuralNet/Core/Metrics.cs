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

        /// Точность бинарной задачи: предсказание = p >= threshold.
        public static float BinaryAccuracy(ReadOnlySpan<float> predicted, ReadOnlySpan<float> target, float threshold = 0.5f)
        {
            SpanGuard.EnsureSameLength(predicted, target);
            int correct = 0;
            for (int i = 0; i < predicted.Length; i++)
            {
                float predictedClass = predicted[i] >= threshold ? 1f : 0f;
                if (predictedClass == target[i]) correct++;
            }

            return (float)correct / predicted.Length;
        }
    }
}
