using System;
using System.Runtime.CompilerServices;

namespace IrisNeuralNet.MathCore
{
    internal static class SpanGuard
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void EnsureSameLength(ReadOnlySpan<float> source, Span<float> destination)
        {
            if (source.Length != destination.Length) ThrowLengthMismatch();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void EnsureSameLength(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> destination)
        {
            if (left.Length != right.Length || left.Length != destination.Length) ThrowLengthMismatch();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void EnsureLength(ReadOnlySpan<float> span, int expectedLength, string parameterName)
        {
            if (span.Length != expectedLength) ThrowUnexpectedLength(parameterName, expectedLength);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ThrowLengthMismatch() =>
            throw new ArgumentException("Source and destination spans must have the same length.");

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ThrowUnexpectedLength(string parameterName, int expectedLength) =>
            throw new ArgumentException($"Expected length {expectedLength}.", parameterName);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void EnsureDivisible(int length, int divisor, string parameterName)
        {
            if (divisor <= 0 || length % divisor != 0)
            {
                ThrowNotDivisible(parameterName, divisor);
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ThrowNotDivisible(string parameterName, int divisor) =>
            throw new ArgumentException($"Length must be a multiple of {divisor}.", parameterName);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void EnsureSameLength(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
        {
            if (left.Length != right.Length)
            {
                ThrowLengthMismatch();
            }
        }
    }
}
