using System;
using System.Linq;
using IrisNeuralNet.Data;
using Xunit;

namespace IrisNeuralNet.Tests;

public sealed class FeatureNormalizerTests
{
    [Fact]
    public void Apply_ProducesZeroMeanUnitStd()
    {
        float[] features = { 1f, 10f, 3f, 12f };
        var normalizer = FeatureNormalizer.Fit(features, featureCount: 2, rowCount: 2);

        normalizer.Apply(features, rowCount: 2);

        Assert.Equal(new float[] { -1f, -1f, 1f, 1f }, features);
    }

    [Fact]
    public void Apply_KeepsConstantFeatureSafe()
    {
        float[] features = { 5f, 1f, 5f, 3f };
        var normalizer = FeatureNormalizer.Fit(features, featureCount: 2, rowCount: 2);

        normalizer.Apply(features, rowCount: 2);

        Assert.Equal(0f, features[0]);
        Assert.Equal(0f, features[2]);
    }

    [Fact]
    public void Constant_ScalesPixelsWithoutMutatingSource()
    {
        float[] pixels = { 0f, 127.5f, 255f };
        var normalizer = FeatureNormalizer.Constant(featureCount: 3, mean: 0f, scale: 1f / 255f);

        float[] scaled = normalizer.Transform(pixels, rowCount: 1);

        Assert.Equal(new float[] { 0f, 0.5f, 1f }, scaled);
        Assert.Equal(255f, pixels[2]);
    }
}

public sealed class DatasetSplitterTests
{
    [Fact]
    public void Split_PreservesAllRowsAndSizes()
    {
        Dataset dataset = CreateIndexedDataset(rowCount: 10);

        var (train, validation) = DatasetSplitter.Split(dataset, validationFraction: 0.2f, seed: 42);

        Assert.Equal(8, train.RowCount);
        Assert.Equal(2, validation.RowCount);

        float[] combinedLabels = new float[train.RowCount + validation.RowCount];
        train.Labels.CopyTo(combinedLabels);
        validation.Labels.CopyTo(combinedLabels.AsSpan(train.RowCount));
        Array.Sort(combinedLabels);

        float[] expected = new float[combinedLabels.Length];
        for (int i = 0; i < expected.Length; i++)
        {
            expected[i] = i;
        }

        Assert.Equal(expected, combinedLabels);
    }

    [Fact]
    public void Split_IsDeterministicForSameSeed()
    {
        Dataset dataset = CreateIndexedDataset(rowCount: 10);

        var (trainA, validationA) = DatasetSplitter.Split(dataset, 0.2f, seed: 7);
        var (trainB, validationB) = DatasetSplitter.Split(dataset, 0.2f, seed: 7);

        Assert.Equal(validationA.Labels.ToArray(), validationB.Labels.ToArray());
        Assert.Equal(trainA.Features.ToArray(), trainB.Features.ToArray());
    }

    private static Dataset CreateIndexedDataset(int rowCount)
    {
        var features = new float[rowCount];
        var labels = new float[rowCount];
        for (int i = 0; i < rowCount; i++)
        {
            features[i] = i;      // признак-«отпечаток» строки
            labels[i] = i;        // метка-«отпечаток» строки
        }

        return new Dataset(features, labels, featureCount: 1, classCount: rowCount);
    }
}

public sealed class OneHotEncoderTests
{
    [Fact]
    public void Encode_ProducesOneHotRows()
    {
        float[] oneHot = OneHotEncoder.Encode(new int[] { 0, 1, 2 }, classCount: 3);

        Assert.Equal(new float[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 }, oneHot);
    }

    [Fact]
    public void Encode_Throws_OnOutOfRangeClass()
    {
        Assert.Throws<ArgumentException>(() => OneHotEncoder.Encode(new int[] { 3 }, classCount: 3));
    }
}
