using System;
using System.IO;
using IrisNeuralNet.Core;
using IrisNeuralNet.Diagnostics;

namespace IrisNeuralNet.Lab.Reporting;

/// <summary>Строка прогресса раз в несколько эпох (аналог verbose=2 в Keras).</summary>
internal sealed class ConsoleProgress : ITrainingObserver
{
    private readonly TextWriter _output;
    private readonly Metric _metric;
    private readonly int _totalEpochs;
    private readonly int _every;

    public ConsoleProgress(TextWriter output, Metric metric, int totalEpochs, int reports = 10)
    {
        _output = output ?? throw new ArgumentNullException(nameof(output));
        _metric = metric ?? throw new ArgumentNullException(nameof(metric));
        _totalEpochs = totalEpochs;
        _every = Math.Max(1, totalEpochs / Math.Max(1, reports));
    }

    public void OnEpochCompleted(in EpochMetrics m)
    {
        if (m.Epoch != 1 && m.Epoch % _every != 0 && m.Epoch != _totalEpochs) return;

        string name = _metric.Name;
        _output.WriteLine(
            $"  epoch {m.Epoch,4}/{_totalEpochs} | loss {MetricFormat.Loss(m.TrainLoss)} | {name} {MetricFormat.Value(_metric, m.TrainMetric),7}"
            + (m.HasValidation
                ? $" | val_loss {MetricFormat.Loss(m.ValidationLoss)} | val_{name} {MetricFormat.Value(_metric, m.ValidationMetric),7}"
                : string.Empty));
    }
}
