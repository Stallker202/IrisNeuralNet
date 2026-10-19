using System;

namespace IrisNeuralNet.Data;

/// <summary>
/// Аффинное преобразование признаков x' = (x − mean) · scale, по признаку.
/// <see cref="Fit"/> — z-score по обучающей выборке (scale = 1/σ); <see cref="Constant"/> —
/// фиксированное масштабирование (например, пиксели ÷255 для MNIST). Оба варианта
/// сериализуются в файл модели одинаково, поэтому загруженная модель сама знает свою предобработку.
/// </summary>
public sealed class FeatureNormalizer
{
    private const float StdFloor = 1e-8f;
    private readonly float[] _means;
    private readonly float[] _scales;

    private FeatureNormalizer(float[] means, float[] scales)
    {
        _means = means;
        _scales = scales;
    }

    public int FeatureCount => _means.Length;
    public ReadOnlySpan<float> Means => _means;
    public ReadOnlySpan<float> Scales => _scales;

    /// <summary>
    /// Z-score: mean и σ считаются только по обучающей выборке — иначе статистики
    /// валидации «протекают» в обучение (data leakage). Константный признак (σ≈0) не масштабируется.
    /// </summary>
    public static FeatureNormalizer Fit(ReadOnlySpan<float> features, int featureCount, int rowCount)
    {
        if (featureCount <= 0) throw new ArgumentOutOfRangeException(nameof(featureCount));
        if (rowCount <= 0) throw new ArgumentOutOfRangeException(nameof(rowCount));
        if (features.Length != rowCount * featureCount)
        {
            throw new ArgumentException("Features length must equal rowCount * featureCount.", nameof(features));
        }

        // double-аккумуляторы: у признаков вроде TAX (Boston) сумма квадратов велика для float.
        var sums = new double[featureCount];
        for (int r = 0; r < rowCount; r++)
        {
            for (int f = 0; f < featureCount; f++) sums[f] += features[r * featureCount + f];
        }

        var means = new float[featureCount];
        var scales = new float[featureCount];
        for (int f = 0; f < featureCount; f++)
        {
            double mean = sums[f] / rowCount;
            double variance = 0;
            for (int r = 0; r < rowCount; r++)
            {
                double diff = features[r * featureCount + f] - mean;
                variance += diff * diff;
            }

            float std = (float)Math.Sqrt(variance / rowCount);
            means[f] = (float)mean;
            scales[f] = std > StdFloor ? 1f / std : 1f;
        }

        return new FeatureNormalizer(means, scales);
    }

    /// <summary>Одинаковое преобразование для всех признаков: x' = (x − mean) · scale.</summary>
    public static FeatureNormalizer Constant(int featureCount, float mean, float scale)
    {
        if (featureCount <= 0) throw new ArgumentOutOfRangeException(nameof(featureCount));
        if (scale == 0f || !float.IsFinite(scale)) throw new ArgumentOutOfRangeException(nameof(scale));

        var means = new float[featureCount];
        var scales = new float[featureCount];
        Array.Fill(means, mean);
        Array.Fill(scales, scale);
        return new FeatureNormalizer(means, scales);
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

    /// <summary>Применяет преобразование на месте.</summary>
    public void Apply(Span<float> features, int rowCount) => Transform(features, features, rowCount);

    /// <summary>Пишет преобразованные признаки в destination (может совпадать с source).</summary>
    public void Transform(ReadOnlySpan<float> source, Span<float> destination, int rowCount)
    {
        int width = _means.Length;
        if (source.Length != rowCount * width || destination.Length != source.Length)
        {
            throw new ArgumentException($"Expected {rowCount} rows of {width} features.");
        }

        for (int r = 0; r < rowCount; r++)
        {
            int offset = r * width;
            for (int f = 0; f < width; f++)
            {
                destination[offset + f] = (source[offset + f] - _means[f]) * _scales[f];
            }
        }
    }

    /// <summary>Новый массив с преобразованными признаками; исходный не меняется.</summary>
    public float[] Transform(ReadOnlySpan<float> source, int rowCount)
    {
        var destination = new float[source.Length];
        Transform(source, destination, rowCount);
        return destination;
    }
}
