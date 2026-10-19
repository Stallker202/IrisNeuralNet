using System;
using System.IO;
using IrisNeuralNet.Data;
using Xunit;

namespace IrisNeuralNet.Tests;

public sealed class DatasetTests
{
    [Fact]
    public void Select_CopiesRowsInRequestedOrder()
    {
        Dataset dataset = Indexed(rowCount: 4, featureCount: 2);

        Dataset selected = dataset.Select(new[] { 3, 1 });

        Assert.Equal(new float[] { 30f, 31f, 10f, 11f }, selected.Features.ToArray());
        Assert.Equal(new float[] { 3f, 1f }, selected.Labels.ToArray());
    }

    [Fact]
    public void Constructor_RejectsSingleClassAndMismatchedRows()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Dataset(new float[2], new float[2], 1, classCount: 1));
        Assert.Throws<ArgumentException>(() => new Dataset(new float[3], new float[2], 1, classCount: 0));
    }

    [Fact]
    public void Split_GivesKerasSizesForBostonHousing()
    {
        Dataset dataset = Indexed(rowCount: 506, featureCount: 1, classCount: 0);

        var (train, test) = DatasetSplitter.Split(dataset, validationFraction: 0.2f, seed: 42);

        Assert.Equal(404, train.RowCount);
        Assert.Equal(102, test.RowCount);
    }

    [Fact]
    public void Split_KeepsIrisSizesStable()
    {
        var (train, validation) = DatasetSplitter.Split(Indexed(150, 1), 0.2f, seed: 42);

        Assert.Equal(120, train.RowCount);
        Assert.Equal(30, validation.RowCount);
    }

    [Fact]
    public void KFold_UsesSequentialBlocksAndKeepsRemainderInTraining()
    {
        Dataset dataset = Indexed(rowCount: 10, featureCount: 1);

        Fold[] folds = KFold.Split(dataset, k: 3);

        Assert.Equal(3, folds.Length);
        Assert.Equal(new float[] { 0f, 1f, 2f }, folds[0].Validation.Labels.ToArray());
        Assert.Equal(new float[] { 3f, 4f, 5f, 6f, 7f, 8f, 9f }, folds[0].Train.Labels.ToArray());
        Assert.Equal(new float[] { 6f, 7f, 8f }, folds[2].Validation.Labels.ToArray());
        Assert.Equal(new float[] { 0f, 1f, 2f, 3f, 4f, 5f, 9f }, folds[2].Train.Labels.ToArray());
    }

    [Fact]
    public void KFold_RejectsTooFewFolds()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => KFold.Split(Indexed(10, 1), k: 1));
    }

    [Fact]
    public void ClassIndices_ThrowsForRegression()
    {
        Assert.Throws<InvalidOperationException>(() => Indexed(3, 1, classCount: 0).ClassIndices());
    }

    [Fact]
    public void CsvRegression_ParsesNumericTarget()
    {
        string path = TempFile("1,2,3.5\r\n4,5,6\n");
        try
        {
            Dataset dataset = CsvDatasetLoader.LoadRegression(path, featureCount: 2);

            Assert.Equal(0, dataset.ClassCount);
            Assert.Equal(new float[] { 1f, 2f, 4f, 5f }, dataset.Features.ToArray());
            Assert.Equal(new float[] { 3.5f, 6f }, dataset.Labels.ToArray());
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void CsvRegression_RejectsNonNumericTarget()
    {
        string path = TempFile("1,2,abc\n");
        try
        {
            FormatException error = Assert.Throws<FormatException>(() => CsvDatasetLoader.LoadRegression(path, 2));
            Assert.Contains("abc", error.Message);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void CsvClassification_ReportsUnknownLabelText()
    {
        string path = TempFile("1,2,Iris-unknown\n");
        try
        {
            FormatException error = Assert.Throws<FormatException>(
                () => CsvDatasetLoader.LoadClassification(path, 2, new[] { "a", "b" }));
            Assert.Contains("Iris-unknown", error.Message);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void BostonLoader_ReadsThirteenFeaturesAndPrice()
    {
        string path = TempFile("0.00632,18,2.31,0,0.538,6.575,65.2,4.09,1,296,15.3,396.9,4.98,24\n");
        try
        {
            Dataset dataset = BostonHousingDataLoader.Load(path);

            Assert.Equal(13, dataset.FeatureCount);
            Assert.Equal(24f, dataset.Labels[0]);
            Assert.Equal(4.98f, dataset.Features[12], precision: 5);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static Dataset Indexed(int rowCount, int featureCount, int classCount = -1)
    {
        var features = new float[rowCount * featureCount];
        var labels = new float[rowCount];
        for (int r = 0; r < rowCount; r++)
        {
            labels[r] = r;
            for (int f = 0; f < featureCount; f++) features[r * featureCount + f] = r * 10 + f;
        }

        return new Dataset(features, labels, featureCount, classCount < 0 ? rowCount : classCount);
    }

    private static string TempFile(string content)
    {
        string path = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".csv");
        File.WriteAllText(path, content);
        return path;
    }
}
