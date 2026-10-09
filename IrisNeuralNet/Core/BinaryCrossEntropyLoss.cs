using IrisNeuralNet.MathCore;
using System;

/// Бинарная кросс-энтропия. В композиции с сигмоидой даёт канонический
/// градиент (p - y) / N — это проверено тестом.

namespace IrisNeuralNet.Core
{
    public sealed class BinaryCrossEntropyLoss : ILossFunction
    {
        private const float ProbabilityFloor = 1e-8f;

        public float Compute(ReadOnlySpan<float> predicted, ReadOnlySpan<float> target, int rowWidth)
        {
            SpanGuard.EnsureSameLength(predicted, target);
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
            SpanGuard.EnsureSameLength(predicted, target);
            SpanGuard.EnsureSameLength(predicted, destination);

            float rows = predicted.Length / rowWidth;
            for (int i = 0; i < predicted.Length; i++)
            {
                float p = ClampProbability(predicted[i]);
                destination[i] = (-(target[i] / p) + (1f - target[i]) / (1f - p)) / rows;
            }
        }

        private static float ClampProbability(float p) =>
            MathF.Min(MathF.Max(p, ProbabilityFloor), 1f - ProbabilityFloor);
    }
}
