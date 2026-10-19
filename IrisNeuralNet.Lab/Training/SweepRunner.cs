using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using IrisNeuralNet.Core;
using IrisNeuralNet.Lab.Experiments;

namespace IrisNeuralNet.Lab.Training;

/// <summary>Сетка sweep: варианты числа скрытых слоёв и нейронов в каждом.</summary>
internal sealed record SweepGrid(int[] LayerOptions, int[] NeuronOptions)
{
    public int Count => LayerOptions.Length * NeuronOptions.Length;
}

internal readonly record struct SweepResult(
    int HiddenLayers,
    int Neurons,
    float TrainMetric,
    float ValidationMetric,
    float ValidationLoss,
    long ElapsedMs);

/// <summary>Пакетное сравнение архитектур «слои × нейроны»; каждый прогон независим и детерминирован.</summary>
internal sealed class SweepRunner
{
    private readonly PreparedData _data;

    public SweepRunner(PreparedData data) => _data = data ?? throw new ArgumentNullException(nameof(data));

    public Metric Metric => _data.Task.Metric;

    /// <param name="template">Оптимизатор, learning rate, batch и эпохи берутся отсюда; архитектура и seed — из сетки.</param>
    public List<SweepResult> Run(SweepGrid grid, ExperimentConfig template, IProgress<SweepResult>? progress = null, CancellationToken token = default)
    {
        if (grid is null) throw new ArgumentNullException(nameof(grid));
        if (template is null) throw new ArgumentNullException(nameof(template));

        var cells = new List<(int Layers, int Neurons, int Index)>();
        foreach (int layers in grid.LayerOptions)
        {
            foreach (int neurons in grid.NeuronOptions)
            {
                cells.Add((layers, neurons, cells.Count));
            }
        }

        var results = new ConcurrentBag<SweepResult>();
        Parallel.ForEach(cells, new ParallelOptions { CancellationToken = token }, cell =>
        {
            var stopwatch = Stopwatch.StartNew();
            ExperimentConfig config = template with
            {
                Name = $"{cell.Layers}x{cell.Neurons}",
                Hidden = ArchitectureParser.Uniform(cell.Layers, cell.Neurons),
                Seed = template.Seed + cell.Index,
            };
            Experiment experiment = Experiment.Create(_data, config);
            experiment.Run(token);
            Evaluation validation = experiment.EvaluateValidation();

            var result = new SweepResult(
                cell.Layers, cell.Neurons, experiment.EvaluateTrain().Metric, validation.Metric, validation.Loss, stopwatch.ElapsedMilliseconds);
            results.Add(result);
            progress?.Report(result);
        });

        return results.OrderBy(r => r.HiddenLayers).ThenBy(r => r.Neurons).ToList();
    }

    /// <summary>
    /// Лучший результат с учётом направления метрики (accuracy — больше, MAE — меньше). Не зависит
    /// от порядка, в котором параллельные прогоны завершились: при равенстве — меньший val_loss,
    /// затем более простая архитектура. Разошедшийся прогон (NaN) считается худшим.
    /// </summary>
    public static SweepResult Best(IEnumerable<SweepResult> results, Metric metric) =>
        results
            .OrderBy(r => float.IsNaN(r.ValidationMetric) ? float.PositiveInfinity : metric.HigherIsBetter ? -r.ValidationMetric : r.ValidationMetric)
            .ThenBy(r => float.IsNaN(r.ValidationLoss) ? float.PositiveInfinity : r.ValidationLoss)
            .ThenBy(r => r.HiddenLayers)
            .ThenBy(r => r.Neurons)
            .First();
}
