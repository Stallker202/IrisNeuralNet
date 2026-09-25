using IrisNeuralNet.Core;
using Xunit;

namespace IrisNeuralNet.Tests;

public sealed class NetworkFactoryTests
{
    [Fact]
    public void Create_BuildsExpectedLayerChain()
    {
        NeuralNetwork network = NetworkFactory.Create(inputDim: 4, new[] { 8, 6 }, classCount: 3, seed: 1);

        Assert.Equal(3, network.LayerCount);
        Assert.Equal(4, network.Layers[0].InputDim);
        Assert.Equal(8, network.Layers[0].OutputDim);
        Assert.Equal(8, network.Layers[1].InputDim);
        Assert.Equal(6, network.Layers[1].OutputDim);
        Assert.Equal(6, network.Layers[2].InputDim);
        Assert.Equal(3, network.Layers[2].OutputDim);
    }

    [Fact]
    public void Create_WithNoHiddenLayers_ProducesSingleSoftmaxLayer()
    {
        NeuralNetwork network = NetworkFactory.Create(inputDim: 4, System.Array.Empty<int>(), classCount: 3, seed: 1);

        Assert.Equal(1, network.LayerCount);
        Assert.Equal(4, network.InputDim);
        Assert.Equal(3, network.OutputDim);
    }

    [Fact]
    public void Create_IsDeterministicForSameSeed()
    {
        NeuralNetwork first = NetworkFactory.Create(4, new[] { 8 }, 3, seed: 42);
        NeuralNetwork second = NetworkFactory.Create(4, new[] { 8 }, 3, seed: 42);
        NeuralNetwork third = NetworkFactory.Create(4, new[] { 8 }, 3, seed: 43);

        Assert.Equal(first.Layers[0].Weights.ToArray(), second.Layers[0].Weights.ToArray());
        Assert.NotEqual(first.Layers[0].Weights.ToArray(), third.Layers[0].Weights.ToArray());
    }
}