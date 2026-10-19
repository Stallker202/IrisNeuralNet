using System;
using IrisNeuralNet.Core;
using IrisNeuralNet.Data;

namespace IrisNeuralNet.Lab.Experiments;

/// <summary>Данные, готовые к обучению: разбиение, нормализация и закодированные цели.</summary>
internal sealed class PreparedData
{
    public required DatasetSpec Spec { get; init; }
    public required FeatureScaling Scaling { get; init; }
    public required FeatureNormalizer? Normalizer { get; init; }

    /// <summary>Исходные (ненормализованные) выборки — для визуализации и проверки сохранённой модели.</summary>
    public required Dataset RawTrain { get; init; }
    public required Dataset RawValidation { get; init; }

    public required float[] TrainFeatures { get; init; }
    public required float[] TrainTargets { get; init; }
    public required float[] ValidationFeatures { get; init; }
    public required float[] ValidationTargets { get; init; }

    public TrainingTask Task => Spec.Task;
    public int FeatureCount => RawTrain.FeatureCount;
    public int ClassCount => RawTrain.ClassCount;
    public int TrainRowCount => RawTrain.RowCount;
    public int ValidationRowCount => RawValidation.RowCount;
    public int RowCount => TrainRowCount + ValidationRowCount;
}

internal static class DataPreparation
{
    public const int DefaultSplitSeed = 42;

    public static PreparedData Prepare(DatasetSpec spec, FeatureScaling scaling, float validationFraction, int splitSeed = DefaultSplitSeed) =>
        Prepare(spec, spec.Load(), scaling, validationFraction, splitSeed);

    /// <summary>Подготовка из уже загруженного источника: сценарии читают файл один раз и пробуют варианты предобработки.</summary>
    public static PreparedData Prepare(
        DatasetSpec spec, DatasetSource source, FeatureScaling scaling, float validationFraction, int splitSeed = DefaultSplitSeed)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));

        (Dataset train, Dataset validation) = source.Test is { } test
            ? (source.Train, test)
            : DatasetSplitter.Split(source.Train, validationFraction, splitSeed);

        return PrepareSplit(spec, train, validation, scaling);
    }

    /// <summary>Подготовка явной пары train/validation (блоки K-Fold).</summary>
    public static PreparedData PrepareSplit(DatasetSpec spec, Dataset train, Dataset validation, FeatureScaling scaling)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));

        FeatureNormalizer? normalizer = scaling.CreateNormalizer(train);
        return new PreparedData
        {
            Spec = spec,
            Scaling = scaling,
            Normalizer = normalizer,
            RawTrain = train,
            RawValidation = validation,
            TrainFeatures = Scale(normalizer, train),
            TrainTargets = spec.Task.EncodeTargets(train.Labels, train.ClassCount),
            ValidationFeatures = Scale(normalizer, validation),
            ValidationTargets = spec.Task.EncodeTargets(validation.Labels, validation.ClassCount),
        };
    }

    private static float[] Scale(FeatureNormalizer? normalizer, Dataset dataset) =>
        normalizer is null ? dataset.Features.ToArray() : normalizer.Transform(dataset.Features, dataset.RowCount);
}
