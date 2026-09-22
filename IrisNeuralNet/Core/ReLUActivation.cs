using IrisNeuralNet.MathCore;
using System;

namespace IrisNeuralNet.Core
{
    public sealed class ReLUActivation : ActivationFunction
    {
        public override void Apply(ReadOnlySpan<float> source, Span<float> destination, int rowWidth)
        {
            SpanGuard.EnsureDivisible(source.Length, rowWidth, nameof(rowWidth));
            Activations.ReLU(source, destination);
        }

        public override void Backward(
            ReadOnlySpan<float> source,
            ReadOnlySpan<float> outputGradient,
            Span<float> destination,
            int rowWidth)
        {
            SpanGuard.EnsureDivisible(source.Length, rowWidth, nameof(rowWidth));
            Activations.ReLUBackward(source, outputGradient, destination);
        }
    }
}
