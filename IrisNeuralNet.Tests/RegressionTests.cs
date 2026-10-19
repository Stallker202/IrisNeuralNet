using System;
using IrisNeuralNet.Core;
using IrisNeuralNet.MathCore;
using IrisNeuralNet.Optimizers;
using Xunit;

namespace IrisNeuralNet.Tests;

public sealed class RegressionTests
{
    [Fact]
    public void LinearActivation_IsIdentityForwardAndBackward()
    {
        var activation = new LinearActivation();
        float[] output = new float[3];
        float[] gradient = new float[3];

        activation.Apply(new float[] { -2f, 0f, 5f }, output, rowWidth: 1);
        activation.Backward(new float[] { -2f, 0f, 5f }, new float[] { 1f, 2f, 3f }, gradient, rowWidth: 1);

        Assert.Equal(new float[] { -2f, 0f, 5f }, output);
        Assert.Equal(new float[] { 1f, 2f, 3f }, gradient);
    }

    [Fact]
    public void MeanSquaredError_MatchesHandValue()
    {
        var loss = new MeanSquaredErrorLoss();

        // ((1 − 3)² + (2 − 2)²) / 2 = 2
        float value = loss.Compute(new float[] { 1f, 2f }, new float[] { 3f, 2f }, rowWidth: 1);

        Assert.Equal(2f, value, precision: 6);
    }

    [Fact]
    public void MeanSquaredError_GradientIsTwiceErrorOverCount()
    {
        var loss = new MeanSquaredErrorLoss();
        float[] gradient = new float[2];

        loss.Backward(new float[] { 1f, 2f }, new float[] { 3f, 2f }, rowWidth: 1, gradient);

        Assert.Equal(new float[] { -2f, 0f }, gradient);
    }

    [Fact]
    public void MeanAbsoluteError_AveragesAbsoluteErrors()
    {
        Assert.Equal(1.5f, Metrics.MeanAbsoluteError(new float[] { 1f, 5f }, new float[] { 2f, 3f }, rowWidth: 1), precision: 6);
    }

    [Fact]
    public void RegressionNetwork_LearnsLinearFunction()
    {
        // y = 2x − 1 на отрезке [−1, 1]: сеть с линейным выходом и MSE должна свести MAE почти к нулю.
        const int rows = 64;
        float[] x = new float[rows];
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++)
        {
            x[i] = -1f + 2f * i / (rows - 1);
            y[i] = 2f * x[i] - 1f;
        }

        NeuralNetwork network = NetworkFactory.Create(1, new[] { 16 }, TrainingTask.Regression, classCount: 0, seed: 3);
        var trainer = new Trainer(network, new RmsPropOptimizer(network, 0.01f), TrainingTask.Regression);

        trainer.Fit(x, y, new TrainingOptions(Epochs: 300, BatchSize: 16, Seed: 1));

        Assert.Equal("mae", trainer.Metric.Name);
        Assert.True(trainer.Evaluate(x, y).Metric < 0.05f, $"mae={trainer.Evaluate(x, y).Metric}");
    }
}
