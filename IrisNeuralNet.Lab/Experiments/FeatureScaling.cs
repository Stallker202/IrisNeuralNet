using System;
using IrisNeuralNet.Data;

namespace IrisNeuralNet.Lab.Experiments;

/// <summary>Предобработка признаков, выбираемая в UI и сценариях.</summary>
internal enum FeatureScaling
{
    /// <summary>Сырые признаки (как в листингах Iris и Sonar).</summary>
    None,

    /// <summary>Z-score по обучающей выборке (Boston Housing).</summary>
    Standardize,

    /// <summary>Деление на 255: пиксели MNIST в [0, 1].</summary>
    Pixels,
}

internal static class FeatureScalingExtensions
{
    public static string DisplayName(this FeatureScaling scaling) =>
        scaling switch
        {
            FeatureScaling.None => "без нормализации",
            FeatureScaling.Standardize => "z-score",
            FeatureScaling.Pixels => "÷255 (пиксели)",
            _ => scaling.ToString(),
        };

    /// <summary>Нормализатор, «обученный» только на train — статистики валидации не протекают.</summary>
    public static FeatureNormalizer? CreateNormalizer(this FeatureScaling scaling, Dataset train) =>
        scaling switch
        {
            FeatureScaling.None => null,
            FeatureScaling.Standardize => FeatureNormalizer.Fit(train.Features, train.FeatureCount, train.RowCount),
            FeatureScaling.Pixels => FeatureNormalizer.Constant(train.FeatureCount, mean: 0f, scale: 1f / 255f),
            _ => throw new ArgumentOutOfRangeException(nameof(scaling), scaling, null),
        };
}
