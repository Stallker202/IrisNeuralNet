using System;
using IrisNeuralNet.MathCore;

namespace IrisNeuralNet.Core;

/// <summary>Сигмоида: выход в (0, 1) — вероятность класса 1 в бинарной задаче.</summary>
public sealed class SigmoidActivation : ActivationFunction
{
    public override ActivationKind Kind => ActivationKind.Sigmoid;

    public override void Apply(ReadOnlySpan<float> source, Span<float> destination, int rowWidth) =>
        Activations.Sigmoid(source, destination);

    public override void Backward(
        ReadOnlySpan<float> source,
        ReadOnlySpan<float> outputGradient,
        Span<float> destination,
        int rowWidth) =>
        Activations.SigmoidBackward(source, outputGradient, destination);
}
