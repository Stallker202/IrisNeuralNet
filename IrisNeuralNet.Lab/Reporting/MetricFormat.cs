using System.Globalization;
using IrisNeuralNet.Core;

namespace IrisNeuralNet.Lab.Reporting;

/// <summary>Единое форматирование значений метрик: accuracy — в процентах, MAE и потери — числом.</summary>
internal static class MetricFormat
{
    public static string Value(Metric metric, float value) =>
        float.IsNaN(value) ? "—"
        : metric.HigherIsBetter ? value.ToString("P1", CultureInfo.InvariantCulture)
        : value.ToString("F3", CultureInfo.InvariantCulture);

    public static string Loss(float value) =>
        float.IsNaN(value) ? "—" : value.ToString("F4", CultureInfo.InvariantCulture);

    /// <summary>Подпись оси/колонки: «Accuracy» или «MAE».</summary>
    public static string Label(Metric metric) =>
        metric.Name == "mae" ? "MAE" : CultureInfo.InvariantCulture.TextInfo.ToTitleCase(metric.Name);
}
