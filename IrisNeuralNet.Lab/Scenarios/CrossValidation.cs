using System;
using System.Collections.Generic;
using System.Linq;
using IrisNeuralNet.Data;
using IrisNeuralNet.Diagnostics;
using IrisNeuralNet.Lab.Experiments;
using IrisNeuralNet.Lab.Reporting;

namespace IrisNeuralNet.Lab.Scenarios;

/// <summary>Результат K-Fold: история каждого блока и усреднённая по блокам валидационная метрика.</summary>
internal sealed record CrossValidationResult(string Name, IReadOnlyList<TrainingHistory> Folds)
{
    /// <summary>Среднее значение валидационной метрики по блокам для каждой эпохи.</summary>
    public float[] AverageValidationMetric()
    {
        int epochs = Folds.Min(f => f.Count);
        var average = new float[epochs];
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            average[epoch] = Folds.Average(f => f.Epochs[epoch].ValidationMetric);
        }

        return average;
    }

    public float[] FinalValidationMetrics() => Folds.Select(f => f.Last.ValidationMetric).ToArray();
}

internal static class CrossValidation
{
    /// <summary>
    /// K-Fold как в методичке Boston Housing: блоки идут подряд, на каждом блоке обучается новая модель
    /// с тем же конфигом; данные должны быть уже перемешаны и (при необходимости) нормализованы.
    /// </summary>
    public static CrossValidationResult Run(
        ScenarioContext context, DatasetSpec spec, Dataset data, ExperimentConfig config, int k, FeatureScaling scaling)
    {
        var folds = new List<TrainingHistory>();
        context.Output.WriteLine($"  {config.Name}, K = {k}:");
        foreach (Fold fold in KFold.Split(data, k))
        {
            PreparedData prepared = DataPreparation.PrepareSplit(spec, fold.Train, fold.Validation, scaling);
            var history = new TrainingHistory();
            Experiment experiment = Experiment.Create(prepared, config, history);
            experiment.Run();
            folds.Add(history);

            EpochMetrics last = history.Last;
            context.Output.WriteLine(
                $"    блок #{fold.Index}: train {fold.Train.RowCount} / val {fold.Validation.RowCount}, " +
                $"val_{experiment.Trainer.Metric.Name} {MetricFormat.Value(experiment.Trainer.Metric, last.ValidationMetric)}");
        }

        return new CrossValidationResult(config.Name, folds);
    }

    public static float StandardDeviation(IReadOnlyList<float> values)
    {
        double mean = values.Average();
        return (float)Math.Sqrt(values.Sum(v => (v - mean) * (v - mean)) / values.Count);
    }
}
