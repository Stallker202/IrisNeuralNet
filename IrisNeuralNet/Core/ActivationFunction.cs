using System;


namespace IrisNeuralNet.Core
{
    /// <summary>
    /// Функция активации слоя. Применяется к пакету (batch) значений.
    /// </summary>
    public abstract class ActivationFunction
    {
        /// <param name="source">Входные значения (batchSize x rowWidth).</param>
        /// <param name="destination">Куда записать результат.</param>
        /// <param name="rowWidth">Ширина строки (нейронов в слое). Поэлементные активации игнорируют его.</param>
        public abstract void Apply(ReadOnlySpan<float> source, Span<float> destination, int rowWidth);

        /// <param name="source">Пре-активация (z) прямого прохода.</param>
        public abstract void Backward(
            ReadOnlySpan<float> source,
            ReadOnlySpan<float> outputGradient,
            Span<float> destination,
            int rowWidth);
    }
}

