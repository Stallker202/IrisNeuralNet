namespace IrisNeuralNet.Diagnostics;

/// <summary>
/// Метрики одной эпохи (аналог строки History в Keras). Значение метрики зависит от задачи:
/// accuracy для классификации, MAE для регрессии (имя — в <see cref="Core.Trainer.Metric"/>).
/// Без валидации ValidationLoss/ValidationMetric равны NaN.
/// </summary>
public readonly record struct EpochMetrics(
    int Epoch,
    float TrainLoss,
    float TrainMetric,
    float ValidationLoss,
    float ValidationMetric)
{
    public bool HasValidation => !float.IsNaN(ValidationLoss);
}
