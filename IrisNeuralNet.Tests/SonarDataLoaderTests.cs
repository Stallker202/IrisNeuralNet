using System;
using System.Globalization;
using System.IO;
using System.Text;
using IrisNeuralNet.Data;
using Xunit;

namespace IrisNeuralNet.Tests;

public sealed class SonarDataLoaderTests
{
    [Fact]
    public void Load_ParsesSixtyFeaturesAndBinaryClasses()
    {
        string path = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".csv");
        var builder = new StringBuilder();
        for (int row = 0; row < 2; row++)
        {
            for (int f = 0; f < 60; f++)
            {
                builder.Append((row * 0.01f + f * 0.001f).ToString(CultureInfo.InvariantCulture));
                builder.Append(',');
            }

            builder.AppendLine(row == 0 ? "R" : "M");
        }

        File.WriteAllText(path, builder.ToString());
        try
        {
            Dataset dataset = SonarDataLoader.Load(path);

            Assert.Equal(2, dataset.RowCount);
            Assert.Equal(60, dataset.FeatureCount);
            Assert.Equal(2, dataset.ClassCount);
            Assert.Equal(new[] { 0f, 1f }, dataset.Labels.ToArray());
            Assert.Equal(0.001f, dataset.Features[1], precision: 6);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_Throws_OnUnknownLabel()
    {
        string path = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".csv");
        var builder = new StringBuilder();
        for (int f = 0; f < 60; f++)
        {
            builder.Append("0.1,");
        }

        builder.AppendLine("X");
        File.WriteAllText(path, builder.ToString());
        try
        {
            Assert.Throws<FormatException>(() => SonarDataLoader.Load(path));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
