using System;
using System.Collections.Generic;
using IrisNeuralNet.Core;
using IrisNeuralNet.Data;
using IrisNeuralNet.Optimizers;

namespace IrisNeuralNet.Lab.Experiments;

/// <summary>Каталог датасетов лабораторных работ.</summary>
internal static class Datasets
{
    public static readonly DatasetSpec Iris = CsvSpec(
        key: "iris",
        name: "Iris",
        fileName: "iris.csv",
        task: TrainingTask.MultiClass,
        load: IrisDataLoader.Load,
        classNames: IrisDataLoader.ClassNames,
        defaults: new ExperimentDefaults(
            Hidden: new[] { 8 }, FeatureScaling.Standardize, ValidationFraction: 0.2f,
            OptimizerKind.Adam, LearningRate: 0.02f, Epochs: 200, BatchSize: 16));

    public static readonly DatasetSpec Sonar = CsvSpec(
        key: "sonar",
        name: "Sonar",
        fileName: "sonar.csv",
        task: TrainingTask.Binary,
        load: SonarDataLoader.Load,
        classNames: SonarDataLoader.ClassNames,
        defaults: new ExperimentDefaults(
            Hidden: new[] { 60 }, FeatureScaling.None, ValidationFraction: 0.1f,
            OptimizerKind.Adam, LearningRate: 0.001f, Epochs: 100, BatchSize: 10));

    public static readonly DatasetSpec BostonHousing = CsvSpec(
        key: "boston",
        name: "Boston Housing",
        fileName: "boston.csv",
        task: TrainingTask.Regression,
        load: BostonHousingDataLoader.Load,
        classNames: null,
        defaults: new ExperimentDefaults(
            Hidden: new[] { 64, 64 }, FeatureScaling.Standardize, ValidationFraction: 0.2f,
            OptimizerKind.RmsProp, LearningRate: 0.001f, Epochs: 100, BatchSize: 16));

    public static readonly DatasetSpec Mnist = new()
    {
        Key = "mnist",
        Name = "MNIST",
        Task = TrainingTask.MultiClass,
        Load = LoadMnist,
        IsAvailable = () => MnistDownloader.FindLocalDirectory() is not null,
        MissingHint = "MNIST ещё не скачан — загрузка начнётся автоматически",
        DownloadAsync = (progress, token) => MnistDownloader.EnsureAsync(progress, token),
        HasPredefinedTestSplit = true,
        InputShape = new[] { MnistDataLoader.ImageSide, MnistDataLoader.ImageSide },
        ClassNames = new[] { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" },
        Defaults = new ExperimentDefaults(
            Hidden: new[] { 256 }, FeatureScaling.Pixels, ValidationFraction: 0f,
            OptimizerKind.Adam, LearningRate: 0.001f, Epochs: 5, BatchSize: 128),
    };

    public static readonly DatasetSpec[] All = { Iris, Sonar, BostonHousing, Mnist };

    public static DatasetSpec? Find(string key) =>
        Array.Find(All, spec => string.Equals(spec.Key, key, StringComparison.OrdinalIgnoreCase));

    private static DatasetSpec CsvSpec(
        string key, string name, string fileName, TrainingTask task, Func<string, Dataset> load,
        IReadOnlyList<string>? classNames, ExperimentDefaults defaults) =>
        new()
        {
            Key = key,
            Name = name,
            Task = task,
            Load = () => new DatasetSource(load(DataFiles.Resolve(fileName)), Test: null),
            IsAvailable = () => DataFiles.Exists(fileName),
            MissingHint = $"нет файла '{fileName}' — положи его рядом с exe",
            ClassNames = classNames,
            Defaults = defaults,
        };

    private static DatasetSource LoadMnist()
    {
        string directory = MnistDownloader.FindLocalDirectory()
            ?? throw new InvalidOperationException("MNIST не найден: сначала скачайте его (MnistDownloader.EnsureAsync).");
        (Dataset train, Dataset test) = MnistDataLoader.LoadDirectory(directory);
        return new DatasetSource(train, test);
    }
}
