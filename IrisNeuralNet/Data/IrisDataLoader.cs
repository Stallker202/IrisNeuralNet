using System.Collections.Generic;

namespace IrisNeuralNet.Data;

/// <summary>Iris (UCI iris.data): 150 строк, 4 признака (см), 3 класса.</summary>
public static class IrisDataLoader
{
    public const int FeatureCount = 4;
    public const int ClassCount = 3;

    private static readonly string[] Labels = { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };

    public static IReadOnlyList<string> ClassNames => Labels;

    public static Dataset Load(string path) => CsvDatasetLoader.LoadClassification(path, FeatureCount, Labels);
}
