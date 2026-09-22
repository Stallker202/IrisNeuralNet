using IrisNeuralNet.Core;
using IrisNeuralNet.MathCore;
using Xunit;

namespace IrisNeuralNet.Tests;

public sealed class CategoricalCrossEntropyLossTests
{
    private readonly CategoricalCrossEntropyLoss _loss = new();

    [Fact]
    public void Compute_MatchesHandComputedValue()
    {
        float loss = _loss.Compute(new float[] { 0.5f, 0.5f }, new float[] { 1f, 0f }, rowWidth: 2);

        Assert.Equal(0.6931472f, loss, precision: 5);
    }

    [Fact]
    public void Compute_AveragesOverBatchRows()
    {
        float[] predicted = { 0.5f, 0.5f, 0.5f, 0.5f };
        float[] target = { 1f, 0f, 1f, 0f };

        float loss = _loss.Compute(predicted, target, rowWidth: 2);

        Assert.Equal(0.6931472f, loss, precision: 5);
    }

    [Fact]
    public void Backward_ScalesByTargetAndBatchSize()
    {
        float[] destination = new float[2];

        _loss.Backward(new float[] { 0.4f, 0.6f }, new float[] { 1f, 0f }, rowWidth: 2, destination);

        Assert.Equal(-2.5f, destination[0], precision: 5);
        Assert.Equal(0f, destination[1], precision: 5);
    }

    [Fact]
    public void Backward_ComposesWithSoftmaxToPredictedMinusTarget()
    {
        float[] logits = { 1f, 3f };
        float[] predicted = new float[2];
        float[] outputGradient = new float[2];
        float[] destination = new float[2];
        Activations.Softmax(logits, predicted);

        _loss.Backward(predicted, new float[] { 0f, 1f }, rowWidth: 2, outputGradient);
        Activations.SoftmaxBackward(logits, outputGradient, destination);

        // dZ = A - Y для связки softmax + cross-entropy
        Assert.Equal(predicted[0] - 0f, destination[0], precision: 5);
        Assert.Equal(predicted[1] - 1f, destination[1], precision: 5);
    }
}