using System;
using IrisNeuralNet.Lab.Experiments;
using IrisNeuralNet.Optimizers;

namespace IrisNeuralNet.Lab.Web;

/// <summary>Проверка настроек прогона на границе HTTP: одна для обучения и для sweep.</summary>
internal static class RequestValidation
{
    public const int MaxEpochs = 5000;
    public const int MaxBatchSize = 4096;
    public const int MaxPatience = 1000;
    public const int Seed = 7;

    public static bool TryCreateConfig(TrainRequest? request, DatasetSpec spec, out ExperimentConfig config, out string error)
    {
        config = null!;
        if (request is null)
        {
            error = "пустой запрос";
            return false;
        }

        if (!ArchitectureParser.TryParse(request.Hidden, out int[] hidden, out error)) return false;

        error = request switch
        {
            _ when !Enum.IsDefined(request.Optimizer) => "неизвестный оптимизатор",
            _ when !IsLearningRate(request.LearningRate) => "learning rate должен быть больше 0 и не больше 10",
            _ when request.Epochs is < 1 or > MaxEpochs => $"эпох — от 1 до {MaxEpochs}",
            _ when request.BatchSize is < 1 or > MaxBatchSize => $"batch size — от 1 до {MaxBatchSize}",
            _ when request.EarlyStoppingPatience is < 0 or > MaxPatience => $"patience — от 0 до {MaxPatience}",
            _ => string.Empty,
        };
        if (error.Length > 0) return false;

        config = new ExperimentConfig(
            Name: $"{spec.Name} [{ArchitectureParser.Format(hidden)}]",
            Hidden: hidden,
            Optimizer: request.Optimizer,
            LearningRate: request.LearningRate,
            Epochs: request.Epochs,
            BatchSize: request.BatchSize,
            Seed: Seed,
            EarlyStoppingPatience: request.EarlyStoppingPatience);
        return true;
    }

    public static bool IsLearningRate(float value) => value > 0f && value <= 10f;
}
