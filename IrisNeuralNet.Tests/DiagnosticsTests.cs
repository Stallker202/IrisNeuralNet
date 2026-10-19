using System;
using IrisNeuralNet.Core;
using IrisNeuralNet.Diagnostics;
using Xunit;

namespace IrisNeuralNet.Tests;

public sealed class DiagnosticsTests
{
    [Fact]
    public void ModelSummary_MatchesKerasParameterCountsForMnistBaseline()
    {
        NeuralNetwork network = NetworkFactory.Create(784, new[] { 256 }, TrainingTask.MultiClass, classCount: 10, seed: 1);

        string summary = ModelSummary.Format(network, new[] { 28, 28 });

        Assert.Contains("flatten (Flatten)", summary);
        Assert.Contains("200,960", summary);   // 784·256 + 256
        Assert.Contains("2,570", summary);     // 256·10 + 10
        Assert.Contains("Total params: 203,530", summary);
        Assert.Equal(203_530, network.ParameterCount);
    }

    [Fact]
    public void ModelSummary_ArchitectureIsCompact()
    {
        NeuralNetwork network = NetworkFactory.Create(60, new[] { 60, 15 }, TrainingTask.Binary, classCount: 2, seed: 1);

        Assert.Equal("60 → 60 → 15 → 1", ModelSummary.Architecture(network));
    }

    [Fact]
    public void TrainingHistory_FindsBestEpochInEitherDirection()
    {
        var history = new TrainingHistory();
        history.OnEpochCompleted(new EpochMetrics(1, 1.0f, 0.5f, 1.2f, 0.4f));
        history.OnEpochCompleted(new EpochMetrics(2, 0.6f, 0.7f, 0.9f, 0.6f));
        history.OnEpochCompleted(new EpochMetrics(3, 0.4f, 0.8f, 1.1f, 0.55f));

        Assert.Equal(2, history.Best(m => m.ValidationLoss, higherIsBetter: false).Epoch);
        Assert.Equal(2, history.Best(m => m.ValidationMetric, higherIsBetter: true).Epoch);
        Assert.Equal(new[] { 1.0f, 0.6f, 0.4f }, history.Series(m => m.TrainLoss));
        Assert.Equal(3, history.Last.Epoch);
    }

    [Fact]
    public void CompositeObserver_ForwardsToEveryObserver()
    {
        var first = new TrainingHistory();
        var second = new TrainingHistory();
        ITrainingObserver composite = new CompositeTrainingObserver(first, second);

        composite.OnEpochCompleted(new EpochMetrics(1, 0f, 0f, float.NaN, float.NaN));

        Assert.Equal(1, first.Count);
        Assert.Equal(1, second.Count);
        Assert.False(second.Last.HasValidation);
    }
}
