using System;
using System.Numerics;

/// Функции активации над плоскими буферами; row-варианты — для батчей.

namespace IrisNeuralNet.MathCore
{
    public static class Activations
    {
        public static void ReLU(ReadOnlySpan<float> source, Span<float> destination)
        {
            SpanGuard.EnsureSameLength(source, destination);

            int i = 0;
            for (; i <= source.Length - Vector<float>.Count; i += Vector<float>.Count)
            {
                Vector.Max(new Vector<float>(source.Slice(i, Vector<float>.Count)), Vector<float>.Zero)
                      .CopyTo(destination.Slice(i));
            }

            for (; i < source.Length; i++) destination[i] = MathF.Max(source[i], 0f);
        }

        public static void Softmax(ReadOnlySpan<float> source, Span<float> destination)
        {
            SpanGuard.EnsureSameLength(source, destination);

            float max = VectorOps.Max(source);
            float sum = 0f;
            for (int i = 0; i < source.Length; i++)
            {
                float exponent = MathF.Exp(source[i] - max);
                destination[i] = exponent;
                sum += exponent;
            }

            VectorOps.Scale(destination, 1f / sum, destination);
        }

        public static void SoftmaxRows(ReadOnlySpan<float> source, Span<float> destination, int columns)
        {
            SpanGuard.EnsureSameLength(source, destination);
            if (columns <= 0 || source.Length % columns != 0)
            {
                throw new ArgumentException("Source length must be a positive multiple of columns.", nameof(columns));
            }

            for (int start = 0; start < source.Length; start += columns)
            {
                Softmax(source.Slice(start, columns), destination.Slice(start, columns));
            }
        }

        /// source — пре-активация (z). destination = outputGradient * (z > 0).
        public static void ReLUBackward(ReadOnlySpan<float> source, ReadOnlySpan<float> outputGradient, Span<float> destination)
        {
            SpanGuard.EnsureSameLength(source, outputGradient, destination);

            int i = 0;
            for (; i <= source.Length - Vector<float>.Count; i += Vector<float>.Count)
            {
                Vector<float> mask = (Vector<float>)Vector.GreaterThan(
                    new Vector<float>(source.Slice(i, Vector<float>.Count)),
                    Vector<float>.Zero);
                Vector.ConditionalSelect(
                        mask,
                        new Vector<float>(outputGradient.Slice(i, Vector<float>.Count)),
                        Vector<float>.Zero)
                    .CopyTo(destination.Slice(i));
            }

            for (; i < source.Length; i++)
            {
                destination[i] = source[i] > 0f ? outputGradient[i] : 0f;
            }
        }

        /// source — пре-активация (z). Произведение Якоби softmax на вектор градиента:
        /// dZ = A * (dA - dot(dA, A)), вычисляется за два прохода без временных буферов.
        public static void SoftmaxBackward(ReadOnlySpan<float> source, ReadOnlySpan<float> outputGradient, Span<float> destination)
        {
            SpanGuard.EnsureSameLength(source, outputGradient, destination);

            float max = VectorOps.Max(source);
            float sum = 0f;
            float weighted = 0f;
            for (int i = 0; i < source.Length; i++)
            {
                float exponent = MathF.Exp(source[i] - max);
                sum += exponent;
                weighted += exponent * outputGradient[i];
            }

            float shift = weighted / sum;
            float inverse = 1f / sum;
            for (int i = 0; i < source.Length; i++)
            {
                float probability = MathF.Exp(source[i] - max) * inverse;
                destination[i] = probability * (outputGradient[i] - shift);
            }
        }

        public static void SoftmaxRowsBackward(
            ReadOnlySpan<float> source,
            ReadOnlySpan<float> outputGradient,
            Span<float> destination,
            int columns)
        {
            SpanGuard.EnsureSameLength(source, outputGradient, destination);
            if (columns <= 0 || source.Length % columns != 0)
            {
                throw new ArgumentException("Source length must be a positive multiple of columns.", nameof(columns));
            }

            for (int start = 0; start < source.Length; start += columns)
            {
                SoftmaxBackward(
                    source.Slice(start, columns),
                    outputGradient.Slice(start, columns),
                    destination.Slice(start, columns));
            }
        }

        /// Численно стойкая сигмоида: ветвление по знаку z защищает от exp-переполнения.
        public static void Sigmoid(ReadOnlySpan<float> source, Span<float> destination)
        {
            SpanGuard.EnsureSameLength(source, destination);
            for (int i = 0; i < source.Length; i++)
            {
                float z = source[i];
                destination[i] = z >= 0f
                    ? 1f / (1f + MathF.Exp(-z))
                    : MathF.Exp(z) / (1f + MathF.Exp(z));
            }
        }

        /// dZ = dA * σ(z) * (1 - σ(z)).
        public static void SigmoidBackward(ReadOnlySpan<float> source, ReadOnlySpan<float> outputGradient, Span<float> destination)
        {
            SpanGuard.EnsureSameLength(source, outputGradient, destination);
            for (int i = 0; i < source.Length; i++)
            {
                float z = source[i];
                float s = z >= 0f
                    ? 1f / (1f + MathF.Exp(-z))
                    : MathF.Exp(z) / (1f + MathF.Exp(z));
                destination[i] = outputGradient[i] * s * (1f - s);
            }
        }
    }
}
