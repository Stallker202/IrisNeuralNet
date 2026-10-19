using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using IrisNeuralNet.Core;
using IrisNeuralNet.Lab.Reporting;
using IrisNeuralNet.Lab.Training;

namespace IrisNeuralNet.Lab.Rendering;

/// <summary>
/// Тепловая карта sweep: колонки — нейроны, строки — число скрытых слоёв. Шкала цвета строится
/// по фактическому диапазону результатов с учётом направления метрики (зелёный — лучше),
/// лучшая ячейка обведена.
/// </summary>
internal static class SweepHeatmapPainter
{
    public static void Draw(SvgBuilder svg, RectangleF bounds, SweepGrid? grid, IReadOnlyList<SweepResult> results, Metric metric)
    {
        svg.Rect(bounds, Theme.ChartBackground);
        svg.Text(8f, 6f, $"Sweep: val {metric.Name} по архитектурам (колонки — нейроны, строки — скрытые слои)", Theme.TextPrimary, TextRole.Title);
        if (grid is null)
        {
            svg.Text(10f, 40f, "Нажми «Запустить sweep»...", Theme.TextMuted);
            return;
        }

        RectangleF plot = RectangleF.FromLTRB(96f, 52f, bounds.Right - 16f, bounds.Bottom - 16f);
        if (plot.Width < 40f || plot.Height < 40f) return;

        float cellWidth = plot.Width / grid.NeuronOptions.Length;
        float cellHeight = plot.Height / grid.LayerOptions.Length;
        DrawHeaders(svg, plot, grid, cellWidth, cellHeight);

        Dictionary<(int Layers, int Neurons), SweepResult> byCell = results.ToDictionary(r => (r.HiddenLayers, r.Neurons));
        SweepResult? best = results.Count > 0 ? SweepRunner.Best(results, metric) : null;
        float worstValue = 0f, bestValue = 0f;
        if (results.Count > 0)
        {
            float min = results.Min(r => r.ValidationMetric);
            float max = results.Max(r => r.ValidationMetric);
            (worstValue, bestValue) = metric.HigherIsBetter ? (min, max) : (max, min);
        }

        for (int r = 0; r < grid.LayerOptions.Length; r++)
        {
            for (int c = 0; c < grid.NeuronOptions.Length; c++)
            {
                float x = plot.Left + c * cellWidth;
                float y = plot.Top + r * cellHeight;
                float w = cellWidth - 2f;
                float h = cellHeight - 2f;
                svg.Rect(x, y, w, h, Theme.EmptyCell);
                if (!byCell.TryGetValue((grid.LayerOptions[r], grid.NeuronOptions[c]), out SweepResult result)) continue;

                svg.Rect(x, y, w, h, Color.FromArgb(210, ScaleColor(result.ValidationMetric, worstValue, bestValue)));
                svg.Text(x + w / 2f, y + h / 2f - 7f, MetricFormat.Value(metric, result.ValidationMetric), Theme.Background, TextRole.Mono, TextAnchor.Middle);
                if (best is { } b && b.HiddenLayers == result.HiddenLayers && b.Neurons == result.Neurons)
                {
                    svg.Rect(x + 1f, y + 1f, w - 2f, h - 2f, fill: null, stroke: Theme.TextPrimary, strokeWidth: 2f);
                }
            }
        }
    }

    private static void DrawHeaders(SvgBuilder svg, RectangleF plot, SweepGrid grid, float cellWidth, float cellHeight)
    {
        for (int c = 0; c < grid.NeuronOptions.Length; c++)
        {
            svg.Text(plot.Left + c * cellWidth + cellWidth / 2f, plot.Top - 18f, grid.NeuronOptions[c].ToString(), Theme.TextMuted, anchor: TextAnchor.Middle);
        }

        for (int r = 0; r < grid.LayerOptions.Length; r++)
        {
            svg.Text(plot.Left - 8f, plot.Top + r * cellHeight + cellHeight / 2f - 7f, $"слоёв: {grid.LayerOptions[r]}", Theme.TextMuted, anchor: TextAnchor.End);
        }
    }

    /// <summary>Худший результат — красный, средний — золотой, лучший — зелёный.</summary>
    private static Color ScaleColor(float value, float worst, float best)
    {
        float range = best - worst;
        float t = MathF.Abs(range) < 1e-9f ? 1f : Math.Clamp((value - worst) / range, 0f, 1f);
        return t < 0.5f
            ? Theme.Mix(Theme.AccentRed, Theme.AccentGold, t * 2f)
            : Theme.Mix(Theme.AccentGold, Theme.AccentGreen, (t - 0.5f) * 2f);
    }
}
