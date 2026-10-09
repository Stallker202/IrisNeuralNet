using System;
using System.Collections.Generic;
using IrisNeuralNet.Core;
using IrisNeuralNet.Data;


namespace IrisNeuralNet.Lab
{
    internal sealed class DatasetSpec
    {
        public required string Name { get; init; }
        public required string FileName { get; init; }
        public required Func<string, IrisDataset> Loader { get; init; }
        public required TaskKind TaskKind { get; init; }
        public required float DefaultValidationFraction { get; init; }
        public required bool DefaultNormalize { get; init; }
        public required int[] DefaultHidden { get; init; }
    }

    internal static class Datasets
    {
        public static readonly DatasetSpec Iris = new()
        {
            Name = "Iris",
            FileName = "iris.csv",
            Loader = IrisDataLoader.Load,
            TaskKind = TaskKind.MultiClass,
            DefaultValidationFraction = 0.2f,
            DefaultNormalize = true,
            DefaultHidden = new[] { 8 },
        };

        public static readonly DatasetSpec Sonar = new()
        {
            Name = "Sonar",
            FileName = "sonar.csv",
            Loader = SonarDataLoader.Load,
            TaskKind = TaskKind.Binary,
            DefaultValidationFraction = 0.1f,
            DefaultNormalize = false,
            DefaultHidden = new[] { 60 },
        };

        public static readonly DatasetSpec[] All = { Iris, Sonar };
    }

    /// Результат подготовки данных для обучения/визуализации.
    internal sealed class PreparedData
    {
        public required TaskKind TaskKind { get; init; }
        public required int FeatureCount { get; init; }
        public required int ClassCount { get; init; }
        public required int RowCount { get; init; }
        public required int TrainRowCount { get; init; }
        public required int ValidationRowCount { get; init; }
        public required float[] TrainFeatures { get; init; }
        public required float[] TrainLabels { get; init; }
        public required int[] TrainClasses { get; init; }
        public required float[] ValidationFeatures { get; init; }
        public required float[] ValidationLabels { get; init; }
        public required float[] RawValidationFeatures { get; init; }
        public required int[] ValidationClasses { get; init; }
        public required FeatureNormalizer? Normalizer { get; init; }
    }

    internal static class DataPreparation
    {
        public static PreparedData Prepare(DatasetSpec spec, float validationFraction, bool normalize, int splitSeed = 42)
        {
            IrisDataset dataset = spec.Loader(spec.FileName);
            (IrisDataset train, IrisDataset validation) =
                DatasetSplitter.Split(dataset, validationFraction, splitSeed);

            float[] trainFeatures = train.Features.ToArray();
            float[] validationFeatures = validation.Features.ToArray();
            float[] rawValidationFeatures = validation.Features.ToArray();

            FeatureNormalizer? normalizer = null;
            if (normalize)
            {
                normalizer = FeatureNormalizer.Fit(trainFeatures, dataset.FeatureCount, train.RowCount);
                normalizer.Apply(trainFeatures, train.RowCount);
                normalizer.Apply(validationFeatures, validation.RowCount);
            }

            return new PreparedData
            {
                TaskKind = spec.TaskKind,
                FeatureCount = dataset.FeatureCount,
                ClassCount = dataset.ClassCount,
                RowCount = dataset.RowCount,
                TrainRowCount = train.RowCount,
                ValidationRowCount = validation.RowCount,
                TrainFeatures = trainFeatures,
                TrainLabels = Encode(train.Classes.ToArray(), spec.TaskKind, dataset.ClassCount),
                TrainClasses = train.Classes.ToArray(),
                ValidationFeatures = validationFeatures,
                ValidationLabels = Encode(validation.Classes.ToArray(), spec.TaskKind, dataset.ClassCount), 
                RawValidationFeatures = rawValidationFeatures,
                ValidationClasses = validation.Classes.ToArray(),
                Normalizer = normalizer,
            };
        }

        private static float[] Encode(int[] classes, TaskKind taskKind, int classCount)
        {
            if (taskKind == TaskKind.Binary)
            {
                var scalar = new float[classes.Length];
                for (int i = 0; i < classes.Length; i++)
                {
                    scalar[i] = classes[i];
                }

                return scalar;
            }

            return OneHotEncoder.Encode(classes, classCount);
        }
    }
}
