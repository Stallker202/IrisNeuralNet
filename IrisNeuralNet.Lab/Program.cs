using System;
using IrisNeuralNet.Core;
using IrisNeuralNet.Data;
using IrisNeuralNet.Optimizers;

namespace IrisNeuralNet.Lab
{
    internal static class Program
    {
        private static void Main()
        {
            Console.WriteLine("Iris classification — low-level C# neural engine");

            IrisDataset dataset = IrisDataLoader.Load("iris.csv");
            (IrisDataset train, IrisDataset validation) = DatasetSplitter.Split(dataset, validationFraction: 0.2f, seed: 42);

            float[] trainFeatures = train.Features.ToArray();
            float[] validationFeatures = validation.Features.ToArray();

            FeatureNormalizer normalizer = FeatureNormalizer.Fit(trainFeatures, IrisDataLoader.FeatureCount, train.RowCount);
            normalizer.Apply(trainFeatures, train.RowCount);
            normalizer.Apply(validationFeatures, validation.RowCount);

            float[] trainLabels = OneHotEncoder.Encode(train.Classes, dataset.ClassCount);
            float[] validationLabels = OneHotEncoder.Encode(validation.Classes, dataset.ClassCount);

            var network = new NeuralNetwork(new[]
            {
            new DenseLayer(IrisDataLoader.FeatureCount, 8, new ReLUActivation(), seed: 1),
            new DenseLayer(8, dataset.ClassCount, new SoftmaxActivation(), seed: 2),
        });

            var observer = new ConsoleTrainingObserver();
            var trainer = new Trainer(
                network,
                new AdamOptimizer(network, learningRate: 0.02f),
                new CategoricalCrossEntropyLoss(),
                observer);

            Console.WriteLine($"rows: {dataset.RowCount} (train {train.RowCount} / val {validation.RowCount})");
            float finalLoss = trainer.Fit(
                trainFeatures, trainLabels,
                validationFeatures, validationLabels,
                epochs: 120, batchSize: 16, seed: 7);

            Console.WriteLine();
            Console.WriteLine($"final train loss : {finalLoss:F4}");
            Console.WriteLine($"train accuracy   : {trainer.EvaluateAccuracy(trainFeatures, trainLabels):P1}");
            Console.WriteLine($"val accuracy     : {trainer.EvaluateAccuracy(validationFeatures, validationLabels):P1}");

            observer.PrintCharts();
        }
    }
}
