using System;
using System.Collections.Generic;

namespace IrisNeuralNet.Diagnostics;

/// <summary>
/// История обучения (аналог объекта History, который возвращает model.fit):
/// метрики каждой эпохи и запросы для анализа переобучения.
/// Не потокобезопасна: для живого UI есть <see cref="MetricsRing"/>.
/// </summary>
public sealed class TrainingHistory : ITrainingObserver
{
    private readonly List<EpochMetrics> _epochs = new();

    public int Count => _epochs.Count;
    public IReadOnlyList<EpochMetrics> Epochs => _epochs;
    public EpochMetrics Last => _epochs.Count > 0 ? _epochs[^1] : throw new InvalidOperationException("History is empty.");

    public void OnEpochCompleted(in EpochMetrics metrics) => _epochs.Add(metrics);

    /// <summary>Ряд значений по эпохам, например <c>history.Series(m => m.ValidationLoss)</c>.</summary>
    public float[] Series(Func<EpochMetrics, float> selector)
    {
        if (selector is null) throw new ArgumentNullException(nameof(selector));

        var values = new float[_epochs.Count];
        for (int i = 0; i < values.Length; i++) values[i] = selector(_epochs[i]);
        return values;
    }

    /// <summary>Эпоха с наилучшим значением ряда (первая при равенстве).</summary>
    public EpochMetrics Best(Func<EpochMetrics, float> selector, bool higherIsBetter)
    {
        if (selector is null) throw new ArgumentNullException(nameof(selector));
        if (_epochs.Count == 0) throw new InvalidOperationException("History is empty.");

        EpochMetrics best = _epochs[0];
        float bestValue = selector(best);
        for (int i = 1; i < _epochs.Count; i++)
        {
            float value = selector(_epochs[i]);
            if (higherIsBetter ? value > bestValue : value < bestValue)
            {
                best = _epochs[i];
                bestValue = value;
            }
        }

        return best;
    }
}
