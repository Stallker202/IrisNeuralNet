using System;
using IrisNeuralNet.Core;
using IrisNeuralNet.Optimizers;
using Xunit;

namespace IrisNeuralNet.Tests;

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
        var trainer = new Trainer(network, new AdamOptimizer(network, 0.05f), TrainingTask.MultiClass);

        float initialLoss = trainer.Evaluate(x, y).Loss;
        float finalLoss = trainer.Fit(x, y, new TrainingOptions(Epochs: 60, BatchSize: 8, Seed: 3)).TrainLoss;

        Assert.True(finalLoss < initialLoss, $"initial={initialLoss}, final={finalLoss}");
        Assert.True(finalLoss < 0.05f, $"final={finalLoss}");
        Assert.True(trainer.Evaluate(x, y).Metric > 0.95f);
    }

    [Fact]
    public void Fit_Throws_WhenRowCountsDiffer()
    {
        var network = new NeuralNetwork(new[] { LayerFactory.Identity(2, new ReLUActivation()) });
        var trainer = new Trainer(network, new SgdOptimizer(0.1f), TrainingTask.MultiClass);

        Assert.Throws<ArgumentException>(
            () => trainer.Fit(new float[4], new float[2], new TrainingOptions(Epochs: 1, BatchSize: 1)));
    }

    [Fact]
    public void Fit_StopsOnCancelledToken()
    {
        var network = new NeuralNetwork(new[] { LayerFactory.Identity(2, new ReLUActivation()) });
        var trainer = new Trainer(network, new SgdOptimizer(0.1f), TrainingTask.MultiClass);
        using var cancellation = new System.Threading.CancellationTokenSource();
        cancellation.Cancel();

        Assert.Throws<System.OperationCanceledException>(
            () => trainer.Fit(new float[4], new float[4], new TrainingOptions(Epochs: 5, BatchSize: 2), cancellation.Token));
    }
}
