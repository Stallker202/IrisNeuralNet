using System;
using IrisNeuralNet.MathCore;

namespace IrisNeuralNet.Core;

/// <summary>
/// Среднеквадратичная ошибка (MSE) для регрессии, как в Keras: среднее по выходам строки,
/// затем по строкам: L = (1 / (N·W)) Σ (p − y)². Градиент: 2 (p − y) / (N·W).
/// Сильнее штрафует большие ошибки, чем MAE.
/// </summary>
public sealed class MeanSquaredErrorLoss : ILossFunction
{
    public float Compute(ReadOnlySpan<float> predicted, ReadOnlySpan<float> target, int rowWidth)
    {
        EnsureBatch(predicted, target, rowWidth);

        float total = 0f;
        for (int i = 0; i < predicted.Length; i++)
        {
            float error = predicted[i] - target[i];
            total += error * error;
        }

        return total / predicted.Length;
    }

    public void Backward(ReadOnlySpan<float> predicted, ReadOnlySpan<float> target, int rowWidth, Span<float> destination)
    {
        EnsureBatch(predicted, target, rowWidth);
        SpanGuard.EnsureSameLength(predicted, destination);

        float scale = 2f / predicted.Length;
        for (int i = 0; i < predicted.Length; i++)
        {
            destination[i] = (predicted[i] - target[i]) * scale;
        }
    }

    private static void EnsureBatch(ReadOnlySpan<float> predicted, ReadOnlySpan<float> target, int rowWidth)
    {
        SpanGuard.EnsureSameLength(predicted, target);
        SpanGuard.EnsureDivisible(predicted.Length, rowWidth, nameof(rowWidth));
    }
}
