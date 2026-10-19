using System;
using IrisNeuralNet.MathCore;

namespace IrisNeuralNet.Core;

/// <summary>
/// Категориальная кросс-энтропия для one-hot меток: L = −(1/N) Σ y·ln p.
/// В композиции с softmax даёт канонический градиент (p − y) / N (закреплено тестом).
/// </summary>
public sealed class CategoricalCrossEntropyLoss : ILossFunction
{
    /// <summary>
    /// Нижняя граница вероятности: защищает ln и деление от нуля, если softmax «схлопнул» класс.
    /// Намеренно крошечная: при p ≥ floor композиция с softmax даёт точное (p − y) / N,
    /// а «крупный» floor (1e-7) гасил бы градиент уверенно ошибочных предсказаний.
    /// </summary>
    private const float ProbabilityFloor = 1e-38f;

    public float Compute(ReadOnlySpan<float> predicted, ReadOnlySpan<float> target, int rowWidth)
    {
        EnsureBatch(predicted, target, rowWidth);

        float total = 0f;
        for (int i = 0; i < predicted.Length; i++)
        {
            if (target[i] != 0f)
            {
                total -= target[i] * MathF.Log(MathF.Max(predicted[i], ProbabilityFloor));
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
