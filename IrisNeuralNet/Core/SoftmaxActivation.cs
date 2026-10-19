using System;
using IrisNeuralNet.MathCore;

namespace IrisNeuralNet.Core;

/// <summary>Softmax по строкам: выход каждой строки — распределение вероятностей классов.</summary>
public sealed class SoftmaxActivation : ActivationFunction
{
    public override ActivationKind Kind => ActivationKind.Softmax;

    public override void Apply(ReadOnlySpan<float> source, Span<float> destination, int rowWidth) =>
        Activations.SoftmaxRows(source, destination, rowWidth);

    public override void Backward(
        ReadOnlySpan<float> source,
        ReadOnlySpan<float> outputGradient,
        Span<float> destination,
        int rowWidth) =>
        Activations.SoftmaxRowsBackward(source, outputGradient, destination, rowWidth);
}
