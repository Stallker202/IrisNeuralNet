using IrisNeuralNet.MathCore;
using System;


namespace IrisNeuralNet.Core
{
    public sealed class SoftmaxActivation : ActivationFunction
    {
        public override void Apply(ReadOnlySpan<float> source, Span<float> destination, int rowWidth) =>
            Activations.SoftmaxRows(source, destination, rowWidth);

        public override void Backward(
        ReadOnlySpan<float> source,
        ReadOnlySpan<float> outputGradient,
        Span<float> destination,
        int rowWidth) =>
        Activations.SoftmaxRowsBackward(source, outputGradient, destination, rowWidth);
    }
}
