using System;
using IrisNeuralNet.MathCore;

namespace IrisNeuralNet.Core;

/// <summary>
/// Линейная (тождественная) активация выходного слоя регрессии: a = z, dZ = dA.
/// Не ограничивает диапазон, поэтому сеть может предсказать любое вещественное число.
/// </summary>
public sealed class LinearActivation : ActivationFunction
{
    public override ActivationKind Kind => ActivationKind.Linear;

    public override void Apply(ReadOnlySpan<float> source, Span<float> destination, int rowWidth)
    {
        SpanGuard.EnsureSameLength(source, destination);
        source.CopyTo(destination);
    }

    public override void Backward(
        ReadOnlySpan<float> source,
        ReadOnlySpan<float> outputGradient,
        Span<float> destination,
        int rowWidth)
    {
        SpanGuard.EnsureSameLength(source, outputGradient, destination);
        outputGradient.CopyTo(destination);
    }
}
