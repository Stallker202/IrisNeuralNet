using System;

namespace IrisNeuralNet.Core;

/// <summary>Единая точка grow-only аренды рабочих буферов: массив только растёт, затем переиспользуется.</summary>
internal static class Buffers
{
    public static Span<float> Rent(ref float[] buffer, int required)
    {
        if (buffer.Length < required)
        {
            buffer = new float[required];
        }

        return buffer.AsSpan(0, required);
    }
}
