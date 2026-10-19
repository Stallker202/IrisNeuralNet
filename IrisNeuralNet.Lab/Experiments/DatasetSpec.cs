using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using IrisNeuralNet.Core;
using IrisNeuralNet.Data;
using IrisNeuralNet.Optimizers;

namespace IrisNeuralNet.Lab.Experiments;

/// <summary>Загруженные данные: train и, если датасет поставляется с готовым разбиением (MNIST), test.</summary>
internal readonly record struct DatasetSource(Dataset Train, Dataset? Test);

/// <summary>Гиперпараметры по умолчанию — как в методичке соответствующей лабораторной.</summary>
internal sealed record ExperimentDefaults(
    int[] Hidden,
    FeatureScaling Scaling,
    float ValidationFraction,
    OptimizerKind Optimizer,
    float LearningRate,
    int Epochs,
    int BatchSize);

/// <summary>Описание датасета для UI и сценариев: откуда грузить, какая задача, какие настройки по умолчанию.</summary>
internal sealed class DatasetSpec
{
    public required string Key { get; init; }
    public required string Name { get; init; }
    public required TrainingTask Task { get; init; }
    public required Func<DatasetSource> Load { get; init; }
    public required Func<bool> IsAvailable { get; init; }
    public required string MissingHint { get; init; }
    public required ExperimentDefaults Defaults { get; init; }

    /// <summary>Скачивание при отсутствии (как keras.datasets.*.load_data); null — файл кладётся вручную.</summary>
    public Func<IProgress<string>?, CancellationToken, Task>? DownloadAsync { get; init; }

    /// <summary>Истина, если test-выборка задана датасетом и доля валидации не используется.</summary>
    public bool HasPredefinedTestSplit { get; init; }

    /// <summary>Исходная форма образца (28×28 для MNIST) — для строки Flatten в сводке модели.</summary>
    public IReadOnlyList<int> InputShape { get; init; } = Array.Empty<int>();

    public IReadOnlyList<string>? ClassNames { get; init; }

    public override string ToString() => Name;
}
