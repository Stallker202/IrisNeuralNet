namespace IrisNeuralNet.Data
{
    public static class SonarDataLoader
    {
        public const int FeatureCount = 60;
        public const int ClassCount = 2;

        private static readonly string[] Labels = { "R", "M" };   // R (rock) -> 0, M (metal) -> 1

        public static IrisDataset Load(string path)
        {
            (float[] features, int[] classes) = CsvDatasetLoader.Load(path, FeatureCount, Labels);
            return new IrisDataset(features, classes, FeatureCount, ClassCount);
        }
    }
}
