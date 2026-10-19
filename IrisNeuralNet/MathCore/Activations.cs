using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace IrisNeuralNet.MathCore;

/// <summary>
/// Функции активации над плоскими буферами; Rows-варианты — для батчей.
/// Backward-методы принимают пре-активацию z прямого прохода.
/// </summary>
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

    /// <summary>destination = outputGradient * (z > 0).</summary>
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

    /// <summary>Численно устойчивый softmax: максимум вычитается до exp, иначе overflow на z ≳ 88.</summary>
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
        SpanGuard.EnsureDivisible(source.Length, columns, nameof(columns));

        for (int start = 0; start < source.Length; start += columns)
        {
            Softmax(source.Slice(start, columns), destination.Slice(start, columns));
        }
    }

    /// <summary>
    /// Произведение Якоби softmax на вектор градиента без матрицы Якоби:
    /// dZ = A ⊙ (dA − Σ dA ⊙ A), два прохода, без временных буферов.
    /// </summary>
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
        SpanGuard.EnsureDivisible(source.Length, columns, nameof(columns));

        for (int start = 0; start < source.Length; start += columns)
        {
            SoftmaxBackward(
                source.Slice(start, columns),
                outputGradient.Slice(start, columns),
                destination.Slice(start, columns));
        }
    }

    public static void Sigmoid(ReadOnlySpan<float> source, Span<float> destination)
    {
        SpanGuard.EnsureSameLength(source, destination);
        for (int i = 0; i < source.Length; i++) destination[i] = Sigmoid(source[i]);
    }

    /// <summary>dZ = dA · σ(z) · (1 − σ(z)).</summary>
    public static void SigmoidBackward(ReadOnlySpan<float> source, ReadOnlySpan<float> outputGradient, Span<float> destination)
    {
        SpanGuard.EnsureSameLength(source, outputGradient, destination);
        for (int i = 0; i < source.Length; i++)
        {
            float s = Sigmoid(source[i]);
            destination[i] = outputGradient[i] * s * (1f - s);
        }
    }

    /// <summary>Численно устойчивая сигмоида: ветвление по знаку z защищает exp от переполнения.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sigmoid(float z)
    {
        if (z >= 0f) return 1f / (1f + MathF.Exp(-z));

        float exponent = MathF.Exp(z);
        return exponent / (1f + exponent);
    }
}
