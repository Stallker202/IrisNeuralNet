using System;

namespace IrisNeuralNet.Core;

/// <summary>Сигнатура метрики: среднее значение по строкам батча.</summary>
public delegate float MetricFunction(ReadOnlySpan<float> predicted, ReadOnlySpan<float> target, int rowWidth);

/// <summary>
/// Именованная метрика обучения (аналог metrics=['accuracy'] / ['mae'] в Keras):
/// функция + направление «лучше». Направление нужно для выбора лучшей эпохи и модели.
/// </summary>
public sealed class Metric
{
    public static readonly Metric Accuracy = new("accuracy", higherIsBetter: true, Metrics.Accuracy);

    public static readonly Metric BinaryAccuracy =
        new("accuracy", higherIsBetter: true, (predicted, target, _) => Metrics.BinaryAccuracy(predicted, target));

    public static readonly Metric MeanAbsoluteError = new("mae", higherIsBetter: false, Metrics.MeanAbsoluteError);

    private readonly MetricFunction _function;

    public Metric(string name, bool higherIsBetter, MetricFunction function)
    {
        Name = string.IsNullOrWhiteSpace(name) ? throw new ArgumentException("Metric name is required.", nameof(name)) : name;
        HigherIsBetter = higherIsBetter;
        _function = function ?? throw new ArgumentNullException(nameof(function));
    }

    public string Name { get; }
    public bool HigherIsBetter { get; }

    public float Compute(ReadOnlySpan<float> predicted, ReadOnlySpan<float> target, int rowWidth) =>
        _function(predicted, target, rowWidth);

    /// <summary>Истина, если candidate строго лучше incumbent с учётом направления метрики.</summary>
    public bool IsBetter(float candidate, float incumbent) =>
        HigherIsBetter ? candidate > incumbent : candidate < incumbent;

    public override string ToString() => Name;
}
