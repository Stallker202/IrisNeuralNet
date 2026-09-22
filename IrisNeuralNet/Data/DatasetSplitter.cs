using IrisNeuralNet.Core;
using System;

namespace IrisNeuralNet.Data
{
    /// <summary>Детерминированное (по seed) разбиение на train/validation.</summary>
    public static class DatasetSplitter
    {
        public static (IrisDataset Train, IrisDataset Validation) Split(
            IrisDataset dataset, float validationFraction, int seed)
        {
            if (validationFraction <= 0f || validationFraction >= 1f)
            {
                throw new ArgumentOutOfRangeException(nameof(validationFraction));
            }

            int rowCount = dataset.RowCount;
            int validationCount = (int)MathF.Round(rowCount * validationFraction);
            if (validationCount == 0 || validationCount == rowCount)
            {
                throw new ArgumentException("Dataset is too small for the requested split.", nameof(dataset));
            }

            int trainCount = rowCount - validationCount;
            var order = new int[rowCount];
            for (int i = 0; i < rowCount; i++) order[i] = i;
            Shuffler.Shuffle(order.AsSpan(), new Random(seed));

            var trainFeatures = new float[trainCount * dataset.FeatureCount];
            var trainClasses = new int[trainCount];
            var validationFeatures = new float[validationCount * dataset.FeatureCount];
            var validationClasses = new int[validationCount];

            for (int i = 0; i < rowCount; i++)
            {
                bool isTrain = i < trainCount;
                int destination = isTrain ? i : i - trainCount;
                Span<float> destinationFeatures = isTrain ? trainFeatures : validationFeatures;
                int[] destinationClasses = isTrain ? trainClasses : validationClasses;

                dataset.Features.Slice(order[i] * dataset.FeatureCount, dataset.FeatureCount)
                    .CopyTo(destinationFeatures.Slice(destination * dataset.FeatureCount));
                destinationClasses[destination] = dataset.Classes[order[i]];
            }

            return (
                new IrisDataset(trainFeatures, trainClasses, dataset.FeatureCount, dataset.ClassCount),
                new IrisDataset(validationFeatures, validationClasses, dataset.FeatureCount, dataset.ClassCount));
        }
    }
}
