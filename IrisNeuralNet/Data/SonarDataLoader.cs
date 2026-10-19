using System.Collections.Generic;

namespace IrisNeuralNet.Data;

/// <summary>Sonar (UCI sonar.all-data): 208 строк, 60 признаков в [0, 1], классы R (камень) → 0, M (металл) → 1.</summary>
public static class SonarDataLoader
{
    public const int FeatureCount = 60;
    public const int ClassCount = 2;

    private static readonly string[] Labels = { "R", "M" };

    public static IReadOnlyList<string> ClassNames => Labels;

    public static Dataset Load(string path) => CsvDatasetLoader.LoadClassification(path, FeatureCount, Labels);
}
