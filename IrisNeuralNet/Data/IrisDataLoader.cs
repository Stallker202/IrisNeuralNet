using System;
using System.Buffers.Text;
using System.IO;
using System.Text;

namespace IrisNeuralNet.Data
{
    /// Читает iris.csv (формат UCI: 4 признака + метка класса, без заголовка)
    /// напрямую из UTF-8 байтов, без промежуточных строк.
    public static class IrisDataLoader
    {
        public const int FeatureCount = 4;
        public const int ClassCount = 3;

        private static readonly string[] Labels = { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };

        public static IrisDataset Load(string path)
        {
            (float[] features, int[] classes) = CsvDatasetLoader.Load(path, FeatureCount, Labels);
            return new IrisDataset(features, classes, FeatureCount, ClassCount);
        }
    }
}