namespace IrisNeuralNet.Core;

/// <summary>Результат оценки на выборке (аналог model.evaluate): значение потерь и метрики.</summary>
public readonly record struct Evaluation(float Loss, float Metric)
{
    /// <summary>«Нет данных» (например, обучение без валидации).</summary>
    public static readonly Evaluation None = new(float.NaN, float.NaN);
}
