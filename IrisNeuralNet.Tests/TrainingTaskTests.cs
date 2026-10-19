using System;
using IrisNeuralNet.Core;
using Xunit;

namespace IrisNeuralNet.Tests;

public sealed class TrainingTaskTests
{
    [Fact]
    public void MultiClass_EncodesOneHotAndUsesSoftmax()
    {
        float[] targets = TrainingTask.MultiClass.EncodeTargets(new float[] { 2f, 0f }, classCount: 3);

        Assert.Equal(new float[] { 0, 0, 1, 1, 0, 0 }, targets);
        Assert.Equal(3, TrainingTask.MultiClass.OutputDim(3));
        Assert.Equal(ActivationKind.Softmax, TrainingTask.MultiClass.CreateOutputActivation().Kind);
        Assert.IsType<CategoricalCrossEntropyLoss>(TrainingTask.MultiClass.CreateLoss());
        Assert.Same(Metric.Accuracy, TrainingTask.MultiClass.Metric);
    }

    [Fact]
    public void MultiClass_RejectsFractionalOrOutOfRangeLabels()
    {
        Assert.Throws<ArgumentException>(() => TrainingTask.MultiClass.EncodeTargets(new float[] { 0.5f }, classCount: 3));
        Assert.Throws<ArgumentException>(() => TrainingTask.MultiClass.EncodeTargets(new float[] { 3f }, classCount: 3));
    }

    [Fact]
    public void Binary_KeepsScalarLabelsAndSingleOutput()
    {
        float[] targets = TrainingTask.Binary.EncodeTargets(new float[] { 1f, 0f, 1f }, classCount: 2);

        Assert.Equal(new float[] { 1f, 0f, 1f }, targets);
        Assert.Equal(1, TrainingTask.Binary.OutputDim(2));
        Assert.Throws<ArgumentOutOfRangeException>(() => TrainingTask.Binary.OutputDim(3));
    }

    [Fact]
    public void Binary_PredictsByHalfThreshold()
    {
        Assert.Equal(1, TrainingTask.Binary.PredictClass(new float[] { 0.8f }, out float positive));
        Assert.Equal(0.8f, positive, precision: 6);
        Assert.Equal(0, TrainingTask.Binary.PredictClass(new float[] { 0.3f }, out float negative));
        Assert.Equal(0.7f, negative, precision: 6);
    }

    [Fact]
    public void MultiClass_PredictsArgMaxWithProbability()
    {
        Assert.Equal(1, TrainingTask.MultiClass.PredictClass(new float[] { 0.2f, 0.7f, 0.1f }, out float confidence));
        Assert.Equal(0.7f, confidence, precision: 6);
    }

    [Fact]
    public void Regression_CopiesTargetsAndUsesLinearMse()
    {
        float[] labels = { 24f, 21.6f };

        float[] targets = TrainingTask.Regression.EncodeTargets(labels, classCount: 0);

        Assert.Equal(labels, targets);
        Assert.NotSame(labels, targets);
        Assert.False(TrainingTask.Regression.IsClassification);
        Assert.Equal(ActivationKind.Linear, TrainingTask.Regression.CreateOutputActivation().Kind);
        Assert.IsType<MeanSquaredErrorLoss>(TrainingTask.Regression.CreateLoss());
        Assert.False(TrainingTask.Regression.Metric.HigherIsBetter);
    }

    [Fact]
    public void Metric_IsBetterRespectsDirection()
    {
        Assert.True(Metric.Accuracy.IsBetter(0.9f, 0.8f));
        Assert.True(Metric.MeanAbsoluteError.IsBetter(2.1f, 2.5f));
        Assert.False(Metric.MeanAbsoluteError.IsBetter(3f, 2.5f));
    }
}
