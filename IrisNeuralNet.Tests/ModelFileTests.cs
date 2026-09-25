using IrisNeuralNet.Core;
using IrisNeuralNet.Data;
using System;
using System.IO;
using Xunit;

namespace IrisNeuralNet.Tests;

public sealed class ModelFileTests
{
    [Fact]
    public void RoundTrip_PreservesWeightsAndOutputs()
    {
        NeuralNetwork network = CreateTwoLayerNetwork();
        string path = Path.GetTempFileName();
        try
        {
            ModelFile.Save(network, null, path);
            LoadedModel loaded = ModelFile.Load(path);

            Assert.Equal(network.Layers[0].Weights.ToArray(), loaded.Network.Layers[0].Weights.ToArray());
            Assert.Equal(network.Layers[1].Biases.ToArray(), loaded.Network.Layers[1].Biases.ToArray());
            Assert.Null(loaded.Normalizer);

            float[] original = new float[3];
            float[] restored = new float[3];
            network.Predict(new float[] { 1f, 2f }, original);
            loaded.Network.Predict(new float[] { 1f, 2f }, restored);
            Assert.Equal(original, restored);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_Throws_OnBadMagic()
    {
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllBytes(path, new byte[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
            Assert.Throws<FormatException>(() => ModelFile.Load(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_Throws_OnTruncatedFile()
    {
        string path = Path.GetTempFileName();
        try
        {
            ModelFile.Save(CreateTwoLayerNetwork(), null, path);
            byte[] bytes = File.ReadAllBytes(path);
            File.WriteAllBytes(path, bytes.AsSpan(0, bytes.Length / 2).ToArray());
            Assert.Throws<FormatException>(() => ModelFile.Load(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void PredictClass_ReturnsMostProbableClass()
    {
        var network = new NeuralNetwork(new[]
        {
            new DenseLayer(new float[] { 1f, -1f, -1f, 1f }, new float[] { 0f, 0f }, new SoftmaxActivation()),
        });

        Assert.Equal(0, network.PredictClass(new float[] { 1f, 0f }));
        Assert.Equal(1, network.PredictClass(new float[] { 0f, 1f }));
    }

    [Fact]
    public void Normalizer_RoundTrip_PreservesParameters()
    {
        float[] features = { 1f, 2f, 3f, 4f };
        FeatureNormalizer normalizer = FeatureNormalizer.Fit(features, featureCount: 2, rowCount: 2);
        string path = Path.GetTempFileName();
        try
        {
            ModelFile.Save(CreateTwoLayerNetwork(), normalizer, path);
            LoadedModel loaded = ModelFile.Load(path);

            FeatureNormalizer? loadedNormalizer = loaded.Normalizer;
            Assert.NotNull(loadedNormalizer);
            Assert.Equal(normalizer.Means.ToArray(), loadedNormalizer!.Means.ToArray());
            Assert.Equal(normalizer.Scales.ToArray(), loadedNormalizer.Scales.ToArray());

            float[] a = { 5f, 6f, 7f, 8f };
            float[] b = { 5f, 6f, 7f, 8f };
            normalizer.Apply(a, rowCount: 2);
            loadedNormalizer.Apply(b, rowCount: 2);
            Assert.Equal(a, b);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static NeuralNetwork CreateTwoLayerNetwork() =>
        new NeuralNetwork(new[]
        {
            new DenseLayer(new float[] { 1f, 2f, 3f, 4f }, new float[] { 0.5f, -0.5f }, new ReLUActivation()),
            new DenseLayer(new float[] { 1f, 0f, 0f, 0f, 1f, 0f }, new float[] { 0f, 0f, 0f }, new SoftmaxActivation()),
        });
}