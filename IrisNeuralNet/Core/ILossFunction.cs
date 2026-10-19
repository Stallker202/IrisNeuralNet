using System;

namespace IrisNeuralNet.Core;

/// <summary>
/// Функция потерь. Контракт: <see cref="Compute"/> возвращает среднее по строкам батча,
/// <see cref="Backward"/> — градиент dL/dA этого среднего (уже поделённый на число строк).
/// Благодаря контракту «среднее по строкам» Trainer может оценивать большие выборки кусками.
/// </summary>
public interface ILossFunction
{
    /// <param name="rowWidth">Ширина строки выхода сети (число нейронов выходного слоя).</param>
    float Compute(ReadOnlySpan<float> predicted, ReadOnlySpan<float> target, int rowWidth);

    void Backward(ReadOnlySpan<float> predicted, ReadOnlySpan<float> target, int rowWidth, Span<float> destination);
}
