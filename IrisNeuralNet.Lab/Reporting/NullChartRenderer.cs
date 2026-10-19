using System.Collections.Generic;
using IrisNeuralNet.Diagnostics;

namespace IrisNeuralNet.Lab.Reporting;

/// <summary>Special Case: графики не сохраняются (флаг --no-charts).</summary>
internal sealed class NullChartRenderer : IChartRenderer
{
    public static readonly NullChartRenderer Instance = new();

    private NullChartRenderer()
    {
    }

    public string? Extension => null;

    public void SaveHistory(string path, string title, TrainingHistory history, string metricLabel)
    {
    }

    public void SaveLines(string path, string title, string yLabel, IReadOnlyList<PlotSeries> series)
    {
    }

    public void SaveDigits(string path, string title, IReadOnlyList<DigitSample> samples)
    {
    }
}
