using System;
using System.Threading;
using IrisNeuralNet.Core;
using IrisNeuralNet.Diagnostics;
using IrisNeuralNet.Optimizers;
using Xunit;

namespace IrisNeuralNet.Tests;

public sealed class TrainerBehaviourTests
{
    [Fact]
    public void Evaluate_InChunksMatchesSingleFullPass()
    {
        // Больше EvaluationChunkRows, причём не кратно ему: проверяем взвешивание последнего куска.
        int rows = Trainer.EvaluationChunkRows * 2 + 37;
        var (x, y) = CreateBlobs(rows, seed: 9);
        NeuralNetwork network = NetworkFactory.Create(2, new[] { 6 }, TrainingTask.MultiClass, classCount: 2, seed: 4);
        var trainer = new Trainer(network, new SgdOptimizer(0.1f), TrainingTask.MultiClass);

        Evaluation chunked = trainer.Evaluate(x, y);

        float[] predicted = new float[rows * 2];
        network.Infer(x, predicted, rows);
        float fullLoss = new CategoricalCrossEntropyLoss().Compute(predicted, y, 2);
        float fullAccuracy = Metrics.Accuracy(predicted, y, 2);
        Assert.Equal(fullLoss, chunked.Loss, precision: 5);
        Assert.Equal(fullAccuracy, chunked.Metric, precision: 5);
    }

    [Fact]
    public void Infer_DoesNotDisturbTrainingCaches()
    {
        NeuralNetwork reference = NetworkFactory.Create(2, new[] { 3 }, TrainingTask.MultiClass, classCount: 2, seed: 8);
        NeuralNetwork probed = NetworkFactory.Create(2, new[] { 3 }, TrainingTask.MultiClass, classCount: 2, seed: 8);
        float[] batch = { 0.5f, -1f, 2f, 0.25f };
        float[] gradient = { 0.1f, -0.1f, -0.2f, 0.2f };

        reference.Forward(batch, new float[4], batchSize: 2);
        reference.Backward(gradient, batchSize: 2);

        probed.Forward(batch, new float[4], batchSize: 2);
        probed.Infer(new float[] { 9f, 9f, -9f, -9f, 3f, 3f }, new float[6], batchSize: 3); // «наблюдатель» между проходами
        probed.Backward(gradient, batchSize: 2);

        for (int i = 0; i < reference.LayerCount; i++)
        {
            Assert.Equal(reference.Layers[i].WeightGradients.ToArray(), probed.Layers[i].WeightGradients.ToArray());
        }
    }

    [Fact]
    public void Fit_StopsMidTrainingWhenTokenIsCancelled()
    {
        var (x, y) = CreateBlobs(64, seed: 2);
        NeuralNetwork network = NetworkFactory.Create(2, new[] { 4 }, TrainingTask.MultiClass, classCount: 2, seed: 1);
        using var cancellation = new CancellationTokenSource();
        var observer = new CancelAfterFirstEpoch(cancellation);
        var trainer = new Trainer(network, new AdamOptimizer(network), TrainingTask.MultiClass, observer);

        Assert.Throws<OperationCanceledException>(
            () => trainer.Fit(x, y, new TrainingOptions(Epochs: 50, BatchSize: 8), cancellation.Token));
        Assert.Equal(1, observer.Epochs);
    }

    [Fact]
    public void Fit_ReportsValidationAndStopsEarly()
    {
        var (x, y) = CreateBlobs(80, seed: 3);
        NeuralNetwork network = NetworkFactory.Create(2, new[] { 8 }, TrainingTask.MultiClass, classCount: 2, seed: 2);
        var history = new TrainingHistory();
        var trainer = new Trainer(network, new AdamOptimizer(network, 0.05f), TrainingTask.MultiClass, history);

        // Валидация с инвертированными метками: чем лучше сеть учит train, тем хуже val_loss —
        // ранняя остановка обязана сработать через patience эпох после лучшей.
        float[] inverted = new float[y.Length];
        for (int i = 0; i < y.Length; i++) inverted[i] = 1f - y[i];
        EpochMetrics last = trainer.Fit(x, y, x, inverted, new TrainingOptions(Epochs: 500, BatchSize: 80, EarlyStoppingPatience: 3));

        Assert.True(last.HasValidation);
        Assert.True(history.Count < 20, $"epochs run: {history.Count}");
        Assert.Equal(history.Last, last);
    }

    [Fact]
    public void Fit_RejectsInvalidOptions()
    {
        NeuralNetwork network = NetworkFactory.Create(2, Array.Empty<int>(), TrainingTask.MultiClass, classCount: 2, seed: 1);
        var trainer = new Trainer(network, new SgdOptimizer(0.1f), TrainingTask.MultiClass);

        Assert.Throws<ArgumentOutOfRangeException>(() => trainer.Fit(new float[2], new float[2], new TrainingOptions(0, 1)));
        Assert.Throws<ArgumentOutOfRangeException>(() => trainer.Fit(new float[2], new float[2], new TrainingOptions(1, 0)));
        Assert.Throws<ArgumentException>(() => trainer.Evaluate(Array.Empty<float>(), Array.Empty<float>()));
    }

    private static (float[] X, float[] Y) CreateBlobs(int rows, int seed)
    {
        var random = new Random(seed);
        float[] x = new float[rows * 2];
        float[] y = new float[rows * 2];
        for (int i = 0; i < rows; i++)
        {
            int cls = i % 2;
            x[i * 2] = cls * 2f + (float)random.NextDouble();
            x[i * 2 + 1] = cls * 2f + (float)random.NextDouble();
            y[i * 2 + cls] = 1f;
        }

        return (x, y);
    }

    private sealed class CancelAfterFirstEpoch : ITrainingObserver
    {
        private readonly CancellationTokenSource _source;

        public CancelAfterFirstEpoch(CancellationTokenSource source) => _source = source;

        public int Epochs { get; private set; }

        public void OnEpochCompleted(in EpochMetrics metrics)
        {
            Epochs++;
            _source.Cancel();
        }
    }
}
