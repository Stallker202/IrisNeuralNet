using IrisNeuralNet.Core;
using IrisNeuralNet.Optimizers;
using System;
using Xunit;

namespace IrisNeuralNet.Tests
{
    public sealed class TrainerTests
    {
        [Fact]
        public void Fit_ConvergesOnSeparableClusters()
        {
            var (x, y) = TestDatasets.CreateTwoClusters(rowsPerClass: 20, seed: 7);
            var network = new NeuralNetwork(new[]
            {
            new DenseLayer(2, 8, new ReLUActivation(), seed: 1),
            new DenseLayer(8, 2, new SoftmaxActivation(), seed: 2),
        });
            var trainer = new Trainer(network, new AdamOptimizer(network, 0.05f), new CategoricalCrossEntropyLoss());

            float initialLoss = trainer.EvaluateLoss(x, y);
            float finalLoss = trainer.Fit(x, y, epochs: 60, batchSize: 8, seed: 3);

            Assert.True(finalLoss < initialLoss, $"initial={initialLoss}, final={finalLoss}");
            Assert.True(finalLoss < 0.05f, $"final={finalLoss}");
            Assert.True(trainer.EvaluateAccuracy(x, y) > 0.95f);
        }

        [Fact]
        public void Fit_Throws_WhenRowCountsDiffer()
        {
            var network = new NeuralNetwork(new[] { LayerFactory.Identity(2, new ReLUActivation()) });
            var trainer = new Trainer(network, new SgdOptimizer(0.1f), new CategoricalCrossEntropyLoss());

            Assert.Throws<ArgumentException>(
                () => trainer.Fit(new float[4], new float[2], epochs: 1, batchSize: 1, seed: 0));
        }
    }
}
