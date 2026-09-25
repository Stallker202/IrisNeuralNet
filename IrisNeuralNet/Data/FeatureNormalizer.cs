using System;

namespace IrisNeuralNet.Data
{
    /// <summary>Стандартизация признаков (z-score); фитится на обучающей выборке.</summary>
    public sealed class FeatureNormalizer
    {
        private const float StdFloor = 1e-8f;
        public ReadOnlySpan<float> Means => _means;
        public ReadOnlySpan<float> Scales => _scales;
        private readonly float[] _means;
        private readonly float[] _scales;

        private FeatureNormalizer(float[] means, float[] scales)
        {
            _means = means;
            _scales = scales;
        }

        public int FeatureCount => _means.Length;

        public static FeatureNormalizer Fit(ReadOnlySpan<float> features, int featureCount, int rowCount)
        {
            if (featureCount <= 0) throw new ArgumentOutOfRangeException(nameof(featureCount));
            if (rowCount <= 0) throw new ArgumentOutOfRangeException(nameof(rowCount));
            if (features.Length != rowCount * featureCount)
            {
                throw new ArgumentException("Features length must equal rowCount * featureCount.", nameof(features));
            }

            var means = new float[featureCount];
            for (int r = 0; r < rowCount; r++)
            {
                for (int f = 0; f < featureCount; f++)
                {
                    means[f] += features[r * featureCount + f];
                }
            }

            var scales = new float[featureCount];
            for (int f = 0; f < featureCount; f++)
            {
                means[f] /= rowCount;
                float variance = 0f;
                for (int r = 0; r < rowCount; r++)
                {
                    float diff = features[r * featureCount + f] - means[f];
                    variance += diff * diff;
                }

                float std = MathF.Sqrt(variance / rowCount);
                scales[f] = std > StdFloor ? 1f / std : 1f;
            }

            return new FeatureNormalizer(means, scales);
        }

        /// <summary>Применяет нормализацию на месте.</summary>
        public void Apply(Span<float> features, int rowCount)
        {
            for (int r = 0; r < rowCount; r++)
            {
                for (int f = 0; f < _means.Length; f++)
                {
                    int index = r * _means.Length + f;
                    features[index] = (features[index] - _means[f]) * _scales[f];
                }
            }
        }

        public static FeatureNormalizer FromParameters(float[] means, float[] scales)
        {
            if (means is null) throw new ArgumentNullException(nameof(means));
            if (scales is null) throw new ArgumentNullException(nameof(scales));
            if (means.Length == 0 || means.Length != scales.Length)
            {
                throw new ArgumentException("Means and scales must be non-empty and of equal length.");
            }

            return new FeatureNormalizer((float[])means.Clone(), (float[])scales.Clone());
        }
    }
}
