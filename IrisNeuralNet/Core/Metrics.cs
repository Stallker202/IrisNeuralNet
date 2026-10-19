using System;
using IrisNeuralNet.MathCore;

namespace IrisNeuralNet.Core;

/// <summary>Метрики качества как чистые функции над span. Все возвращают среднее по строкам.</summary>
public static class Metrics
{
    /// <summary>Доля строк, где argmax предсказания совпал с argmax one-hot метки.</summary>
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

    /// <summary>Точность бинарной задачи: класс 1, если p ≥ threshold.</summary>
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

    /// <summary>Средняя абсолютная ошибка (MAE) — в единицах целевой переменной, удобна для интерпретации.</summary>
    public static float MeanAbsoluteError(ReadOnlySpan<float> predicted, ReadOnlySpan<float> target, int rowWidth)
    {
        SpanGuard.EnsureSameLength(predicted, target);
        SpanGuard.EnsureDivisible(predicted.Length, rowWidth, nameof(rowWidth));

        float total = 0f;
        for (int i = 0; i < predicted.Length; i++) total += MathF.Abs(predicted[i] - target[i]);
        return total / predicted.Length;
    }

    public static int ArgMax(ReadOnlySpan<float> row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++)
        {
            if (row[i] > row[best]) best = i;
        }

        return best;
    }
}
