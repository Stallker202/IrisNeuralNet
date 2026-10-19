using System;
using IrisNeuralNet.MathCore;

namespace IrisNeuralNet.Core;

public sealed class ReLUActivation : ActivationFunction
{
    public override ActivationKind Kind => ActivationKind.Relu;

    public override void Apply(ReadOnlySpan<float> source, Span<float> destination, int rowWidth) =>
        Activations.ReLU(source, destination);

    public override void Backward(
        ReadOnlySpan<float> source,
        ReadOnlySpan<float> outputGradient,
        Span<float> destination,
        int rowWidth) =>
        Activations.ReLUBackward(source, outputGradient, destination);
}
