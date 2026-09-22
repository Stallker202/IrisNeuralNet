using System;


namespace IrisNeuralNet.Core
{
    public interface ILossFunction
    {
        /// <param name="rowWidth">Ширина строки (число классов).</param>
        float Compute(ReadOnlySpan<float> predicted, ReadOnlySpan<float> target, int rowWidth);

        void Backward(ReadOnlySpan<float> predicted, ReadOnlySpan<float> target, int rowWidth, Span<float> destination);
    }
}
