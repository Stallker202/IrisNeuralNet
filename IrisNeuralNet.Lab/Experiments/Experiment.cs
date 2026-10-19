using System;
using System.Threading;
using IrisNeuralNet.Core;
using IrisNeuralNet.Diagnostics;
using IrisNeuralNet.Optimizers;

namespace IrisNeuralNet.Lab.Experiments;

/// <summary>Конфигурация одного прогона: архитектура и гиперпараметры (аналог model + compile + fit).</summary>
internal sealed record ExperimentConfig(
    string Name,
    int[] Hidden,
    OptimizerKind Optimizer,
    float LearningRate,
    int Epochs,
    int BatchSize,
    int Seed = 7,
    int EarlyStoppingPatience = 0)
{
    public static ExperimentConfig FromDefaults(string name, ExperimentDefaults defaults) =>
        new(name, defaults.Hidden, defaults.Optimizer, defaults.LearningRate, defaults.Epochs, defaults.BatchSize);

    public TrainingOptions TrainingOptions => new(Epochs, BatchSize, Seed, EarlyStoppingPatience);
}

/// <summary>
/// Собранная модель, готовая к обучению: единственное место, где из данных и конфига
/// создаются сеть, оптимизатор и тренер (общее для UI, sweep и консольных сценариев).
/// </summary>
internal sealed class Experiment
{
    private Experiment(PreparedData data, ExperimentConfig config, NeuralNetwork network, IOptimizer optimizer, Trainer trainer)
    {
        Data = data;
        Config = config;
        Network = network;
        Optimizer = optimizer;
        Trainer = trainer;
    }

    public PreparedData Data { get; }
    public ExperimentConfig Config { get; }
    public NeuralNetwork Network { get; }
    public IOptimizer Optimizer { get; }
    public Trainer Trainer { get; }

    public static Experiment Create(PreparedData data, ExperimentConfig config, ITrainingObserver? observer = null)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (config is null) throw new ArgumentNullException(nameof(config));

        NeuralNetwork network = NetworkFactory.Create(data.FeatureCount, config.Hidden, data.Task, data.ClassCount, config.Seed);
        IOptimizer optimizer = OptimizerFactory.Create(config.Optimizer, network, config.LearningRate);
        var trainer = new Trainer(network, optimizer, data.Task, observer);
        return new Experiment(data, config, network, optimizer, trainer);
    }

    /// <summary>Обучение с валидацией на отложенной выборке; возвращает метрики последней эпохи.</summary>
    public EpochMetrics Run(CancellationToken token = default) =>
        Trainer.Fit(Data.TrainFeatures, Data.TrainTargets, Data.ValidationFeatures, Data.ValidationTargets, Config.TrainingOptions, token);

    public Evaluation EvaluateTrain() => Trainer.Evaluate(Data.TrainFeatures, Data.TrainTargets);

    public Evaluation EvaluateValidation() => Trainer.Evaluate(Data.ValidationFeatures, Data.ValidationTargets);
}
