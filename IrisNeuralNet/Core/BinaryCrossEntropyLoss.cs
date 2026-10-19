using System;
using IrisNeuralNet.MathCore;

namespace IrisNeuralNet.Core;

/// <summary>
/// Бинарная кросс-энтропия: L = −(1/N) Σ [y·ln p + (1 − y)·ln(1 − p)].
/// В композиции с сигмоидой даёт канонический градиент (p − y) / N (закреплено тестом).
/// </summary>
public sealed class BinaryCrossEntropyLoss : ILossFunction
{
    /// <summary>
    /// Граница отсечения p в [floor, 1 − floor]. Не меньше машинного эпсилона float (~1.2e-7):
    /// при 1e-8 выражение 1 − floor округлялось до 1f, и ln(1 − p) давал −∞ (NaN в градиенте).
    /// </summary>
    private const float ProbabilityFloor = 1e-7f;

    public float Compute(ReadOnlySpan<float> predicted, ReadOnlySpan<float> target, int rowWidth)
    {
        EnsureBatch(predicted, target, rowWidth);

        float total = 0f;
        for (int i = 0; i < predicted.Length; i++)
        {
            float p = ClampProbability(predicted[i]);
            total -= target[i] * MathF.Log(p) + (1f - target[i]) * MathF.Log(1f - p);
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
            float p = ClampProbability(predicted[i]);
            destination[i] = (-(target[i] / p) + (1f - target[i]) / (1f - p)) / rowCount;
        }
    }

    private static float ClampProbability(float p) =>
        MathF.Min(MathF.Max(p, ProbabilityFloor), 1f - ProbabilityFloor);

    private static void EnsureBatch(ReadOnlySpan<float> predicted, ReadOnlySpan<float> target, int rowWidth)
    {
        SpanGuard.EnsureSameLength(predicted, target);
        SpanGuard.EnsureDivisible(predicted.Length, rowWidth, nameof(rowWidth));
    }
}
