using System;
using IrisNeuralNet.Core;
using IrisNeuralNet.Diagnostics;
using IrisNeuralNet.Lab.Experiments;

namespace IrisNeuralNet.Lab.Scenarios;

/// <summary>Итог одного эксперимента: история, финальные оценки и время.</summary>
internal sealed record ExperimentResult(
    Experiment Experiment,
    TrainingHistory History,
    Evaluation Train,
    Evaluation Validation,
    TimeSpan Elapsed)
{
    public ExperimentConfig Config => Experiment.Config;
    public Metric Metric => Experiment.Trainer.Metric;
    public string Architecture => ModelSummary.Architecture(Experiment.Network);

    /// <summary>Эпоха с лучшим значением валидационной метрики.</summary>
    public EpochMetrics BestValidation => History.Best(m => m.ValidationMetric, Metric.HigherIsBetter);

    /// <summary>Эпоха минимума val_loss — после неё начинается переобучение.</summary>
    public EpochMetrics MinValidationLoss => History.Best(m => m.ValidationLoss, higherIsBetter: false);
}
