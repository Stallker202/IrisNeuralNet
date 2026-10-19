using System;

namespace IrisNeuralNet.Data;

/// <summary>
/// Табличный датасет: матрица признаков [RowCount x FeatureCount] и по одной метке на строку.
/// Метка — индекс класса (0..ClassCount−1) для классификации или значение для регрессии
/// (ClassCount = 0). Кодирование меток в цели сети — дело <see cref="Core.TrainingTask"/>.
/// </summary>
public readonly struct Dataset
{
    private readonly float[] _features;
    private readonly float[] _labels;

    public Dataset(float[] features, float[] labels, int featureCount, int classCount)
    {
        if (features is null) throw new ArgumentNullException(nameof(features));
        if (labels is null) throw new ArgumentNullException(nameof(labels));
        if (featureCount <= 0) throw new ArgumentOutOfRangeException(nameof(featureCount));
        if (classCount < 0 || classCount == 1) throw new ArgumentOutOfRangeException(nameof(classCount), "Use 0 for regression or ≥ 2 classes.");
        if (labels.Length == 0 || features.Length != labels.Length * featureCount)
        {
            throw new ArgumentException("Features and labels must describe the same non-zero row count.", nameof(features));
        }

        _features = features;
        _labels = labels;
        FeatureCount = featureCount;
        ClassCount = classCount;
    }

    public int RowCount => _labels.Length;
    public int FeatureCount { get; }

    /// <summary>Число классов; 0 — регрессия.</summary>
    public int ClassCount { get; }

    public bool IsClassification => ClassCount > 0;
    public ReadOnlySpan<float> Features => _features;
    public ReadOnlySpan<float> Labels => _labels;

    public ReadOnlySpan<float> Row(int index) => _features.AsSpan(index * FeatureCount, FeatureCount);

    /// <summary>Новый датасет из строк в указанном порядке (копии строк, исходный не мутируется).</summary>
    public Dataset Select(ReadOnlySpan<int> rows)
    {
        var features = new float[rows.Length * FeatureCount];
        var labels = new float[rows.Length];
        for (int i = 0; i < rows.Length; i++)
        {
            Row(rows[i]).CopyTo(features.AsSpan(i * FeatureCount));
            labels[i] = _labels[rows[i]];
        }

        return new Dataset(features, labels, FeatureCount, ClassCount);
    }

    /// <summary>Новый датасет из непрерывного диапазона строк [start, start + count).</summary>
    public Dataset Slice(int start, int count) =>
        new(
            _features.AsSpan(start * FeatureCount, count * FeatureCount).ToArray(),
            _labels.AsSpan(start, count).ToArray(),
            FeatureCount,
            ClassCount);

    /// <summary>Индексы классов как int (для визуализации); только для классификации.</summary>
    public int[] ClassIndices()
    {
        if (!IsClassification) throw new InvalidOperationException("Regression dataset has no classes.");

        var classes = new int[RowCount];
        for (int i = 0; i < classes.Length; i++) classes[i] = (int)_labels[i];
        return classes;
    }
}
