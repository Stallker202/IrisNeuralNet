using IrisNeuralNet.Core;
using IrisNeuralNet.MathCore;
using Xunit;

namespace IrisNeuralNet.Tests;

public sealed class BinaryTaskTests
{
    [Fact]
    public void Sigmoid_AtZero_IsHalf()
    {
        float[] result = new float[1];
        Activations.Sigmoid(new float[] { 0f }, result);
        Assert.Equal(0.5f, result[0], precision: 6);
    }

    [Fact]
    public void Sigmoid_IsBoundedForExtremeInputs()
    {
        float[] result = new float[2];
        Activations.Sigmoid(new float[] { 100f, -100f }, result);
        Assert.Equal(1f, result[0], precision: 6);
        Assert.Equal(0f, result[1], precision: 6);
    }

    [Fact]
    public void SigmoidBackward_AtZero_IsQuarter()
    {
        float[] result = new float[1];
        Activations.SigmoidBackward(new float[] { 0f }, new float[] { 1f }, result);
        Assert.Equal(0.25f, result[0], precision: 6);
    }

    [Fact]
    public void BinaryCrossEntropy_MatchesHandValue()
    {
        var loss = new BinaryCrossEntropyLoss();
        float value = loss.Compute(new float[] { 0.5f }, new float[] { 1f }, rowWidth: 1);
        Assert.Equal(0.6931472f, value, precision: 5);
    }

    [Fact]
    public void SigmoidPlusBinaryCrossEntropy_GivesPredictedMinusTarget()
    {
        // Композиция backward должна давать канонический градиент (p - y) / N.
        var loss = new BinaryCrossEntropyLoss();
        float[] z = { 0f };
        float[] p = new float[1];
        Activations.Sigmoid(z, p);

        float[] dA = new float[1];
        loss.Backward(p, new float[] { 1f }, rowWidth: 1, dA);

        float[] dZ = new float[1];
        Activations.SigmoidBackward(z, dA, dZ);

        Assert.Equal(p[0] - 1f, dZ[0], precision: 5);
    }

    [Fact]
    public void BinaryAccuracy_CountsThresholdedMatches()
    {
        Assert.Equal(1f, Metrics.BinaryAccuracy(new float[] { 0.6f, 0.4f }, new float[] { 1f, 0f }));
        Assert.Equal(0.5f, Metrics.BinaryAccuracy(new float[] { 0.6f, 0.6f }, new float[] { 1f, 0f }));
    }
}