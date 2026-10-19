using System;

namespace IrisNeuralNet.Core;

/// <summary>
/// Функция активации слоя. Применяется к пакету (batch) значений.
/// Реализации не хранят состояния, поэтому потокобезопасны и могут разделяться слоями.
/// </summary>
public abstract class ActivationFunction
{
    public abstract ActivationKind Kind { get; }

    /// <param name="source">Пре-активация z (batchSize x rowWidth).</param>
    /// <param name="destination">Куда записать результат; может совпадать с source.</param>
    /// <param name="rowWidth">Ширина строки (нейронов в слое). Поэлементные активации её игнорируют.</param>
    public abstract void Apply(ReadOnlySpan<float> source, Span<float> destination, int rowWidth);

    /// <param name="source">Пре-активация z прямого прохода.</param>
    /// <param name="outputGradient">dL/dA.</param>
    /// <param name="destination">Куда записать dL/dZ.</param>
    public abstract void Backward(
        ReadOnlySpan<float> source,
        ReadOnlySpan<float> outputGradient,
        Span<float> destination,
        int rowWidth);
}
