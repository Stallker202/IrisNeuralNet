using System;

namespace IrisNeuralNet.Data
{
    /// <summary>Датасет: матрица признаков [RowCount x FeatureCount] и индексы классов.</summary>
    public readonly struct IrisDataset
    {
        private readonly float[] _features;
        private readonly int[] _classes;

        public IrisDataset(float[] features, int[] classes, int featureCount, int classCount)
        {
            if (features is null) throw new ArgumentNullException(nameof(features));
            if (classes is null) throw new ArgumentNullException(nameof(classes));
            if (featureCount <= 0) throw new ArgumentOutOfRangeException(nameof(featureCount));
            if (classCount <= 0) throw new ArgumentOutOfRangeException(nameof(classCount));
            if (classes.Length == 0 || features.Length != classes.Length * featureCount)
            {
                throw new ArgumentException("Features and classes must describe the same row count.", nameof(features));
            }

            _features = features;
            _classes = classes;
            FeatureCount = featureCount;
            ClassCount = classCount;
        }

        public int RowCount => _classes.Length;
        public int FeatureCount { get; }
        public int ClassCount { get; }
        public ReadOnlySpan<float> Features => _features;
        public ReadOnlySpan<int> Classes => _classes;
    }
}
