using System.Collections.Generic;

namespace IrisNeuralNet.Data;

/// <summary>
/// Boston Housing: 506 строк, 13 признаков (CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX,
/// PTRATIO, B, LSTAT) и цель MEDV — медианная цена дома в тысячах долларов. Формат: CSV без заголовка.
/// </summary>
public static class BostonHousingDataLoader
{
    public const int FeatureCount = 13;

    private static readonly string[] Names =
    {
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
    };

    public static IReadOnlyList<string> FeatureNames => Names;

    public static Dataset Load(string path) => CsvDatasetLoader.LoadRegression(path, FeatureCount);
}
