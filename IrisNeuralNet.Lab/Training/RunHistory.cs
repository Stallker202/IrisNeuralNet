using System.Collections.Generic;
using IrisNeuralNet.Diagnostics;

namespace IrisNeuralNet.Lab.Training;

/// <summary>Четыре ряда одного прогона: потери и метрика на train/validation.</summary>
internal sealed class RunSeries
{
    public List<float> TrainLoss { get; } = new();
    public List<float> ValidationLoss { get; } = new();
    public List<float> TrainMetric { get; } = new();
    public List<float> ValidationMetric { get; } = new();
    public int Count => TrainLoss.Count;

    public void Append(in EpochMetrics metrics)
    {
        TrainLoss.Add(metrics.TrainLoss);
        ValidationLoss.Add(metrics.ValidationLoss);
        TrainMetric.Add(metrics.TrainMetric);
        ValidationMetric.Add(metrics.ValidationMetric);
    }

    public void CopyFrom(RunSeries other)
    {
        Replace(TrainLoss, other.TrainLoss);
        Replace(ValidationLoss, other.ValidationLoss);
        Replace(TrainMetric, other.TrainMetric);
        Replace(ValidationMetric, other.ValidationMetric);
    }

    public void Clear()
    {
        TrainLoss.Clear();
        ValidationLoss.Clear();
        TrainMetric.Clear();
        ValidationMetric.Clear();
    }

    private static void Replace(List<float> target, List<float> source)
    {
        target.Clear();
        target.AddRange(source);
    }
}

/// <summary>Текущий прогон и предыдущий (для оверлея на графиках); забирает новые эпохи из MetricsRing.</summary>
internal sealed class RunHistory
{
    private readonly List<EpochMetrics> _snapshot = new();

    public RunSeries Current { get; } = new();
    public RunSeries Previous { get; } = new();
    public EpochMetrics Latest { get; private set; }
    public bool HasPrevious => Previous.Count > 1;

    /// <summary>Переносит в историю эпохи, появившиеся в кольце с прошлого опроса.</summary>
    public void Consume(MetricsRing ring)
    {
        ring.CopyTo(_snapshot);
        for (int i = Current.Count; i < _snapshot.Count; i++)
        {
            Current.Append(_snapshot[i]);
        }

        Latest = _snapshot.Count > 0 ? _snapshot[^1] : default;
    }

    /// <summary>Начало нового прогона; текущий становится «прошлым», если нужен оверлей.</summary>
    public void StartNewRun(bool keepPrevious)
    {
        if (keepPrevious && Current.Count > 1) Previous.CopyFrom(Current);
        Current.Clear();
        Latest = default;
    }

    public void Reset()
    {
        Current.Clear();
        Previous.Clear();
        Latest = default;
    }
}
