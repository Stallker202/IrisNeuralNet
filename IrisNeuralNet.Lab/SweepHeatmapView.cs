using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Windows.Forms;
using IrisNeuralNet.Core;
using IrisNeuralNet.Lab.Reporting;

namespace IrisNeuralNet.Lab;

/// <summary>
/// Тепловая карта sweep: колонки — нейроны, строки — число скрытых слоёв. Шкала цвета строится
/// по фактическому диапазону результатов с учётом направления метрики (зелёный — лучше).
/// </summary>
internal sealed class SweepHeatmapView : PictureBox
{
    private readonly Dictionary<(int Layers, int Neurons), SweepResult> _results = new();
    private SweepGrid? _grid;
    private Metric _metric = Metric.Accuracy;
    private SweepResult? _best;
    private float _worstValue;
    private float _bestValue;

    public SweepHeatmapView()
    {
        SetStyle(ControlStyles.OptimizedDoubleBuffer | ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint, true);
        UpdateStyles();
    }

    public void UpdateData(IReadOnlyList<SweepResult> results, SweepGrid grid, Metric metric)
    {
        _results.Clear();
        _grid = grid;
        _metric = metric;
        foreach (SweepResult r in results) _results[(r.HiddenLayers, r.Neurons)] = r;

        _best = results.Count > 0 ? SweepRunner.Best(results, metric) : null;
        if (results.Count > 0)
        {
            float min = results.Min(r => r.ValidationMetric);
            float max = results.Max(r => r.ValidationMetric);
            (_worstValue, _bestValue) = metric.HigherIsBetter ? (min, max) : (max, min);
        }
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        Graphics g = e.Graphics;
        g.SmoothingMode = SmoothingMode.AntiAlias;
        using (var background = new SolidBrush(UiTheme.ChartBackground))
        {
            g.FillRectangle(background, ClientRectangle);
        }

        using var titleBrush = new SolidBrush(UiTheme.TextPrimary);
        g.DrawString($"Sweep: val {_metric.Name} по архитектурам (колонки — нейроны, строки — скрытые слои)", UiTheme.TitleFont, titleBrush, 8, 6);

        SweepGrid? grid = _grid;
        if (grid is null)
        {
            using var waiting = new SolidBrush(UiTheme.TextMuted);
            g.DrawString("Нажми «Запустить sweep»...", UiTheme.UiFont, waiting, 10, 40);
            return;
        }

        Rectangle plot = Rectangle.FromLTRB(96, 34, ClientRectangle.Right - 16, ClientRectangle.Bottom - 16);
        if (plot.Width < 40 || plot.Height < 40) return;

        float cellWidth = plot.Width / (float)grid.NeuronOptions.Length;
        float cellHeight = plot.Height / (float)grid.LayerOptions.Length;
        DrawHeaders(g, plot, grid, cellWidth, cellHeight);

        using var empty = new SolidBrush(Color.FromArgb(26, 28, 33));
        using var textBrush = new SolidBrush(UiTheme.Background);
        using var frame = new Pen(UiTheme.TextPrimary, 2f);
        for (int r = 0; r < grid.LayerOptions.Length; r++)
        {
            for (int c = 0; c < grid.NeuronOptions.Length; c++)
            {
                float x = plot.Left + c * cellWidth;
                float y = plot.Top + r * cellHeight;
                float w = cellWidth - 2;
                float h = cellHeight - 2;
                g.FillRectangle(empty, x, y, w, h);
                if (!_results.TryGetValue((grid.LayerOptions[r], grid.NeuronOptions[c]), out SweepResult result)) continue;

                using (var fill = new SolidBrush(Color.FromArgb(210, ScaleColor(result.ValidationMetric))))
                {
                    g.FillRectangle(fill, x, y, w, h);
                }

                string text = MetricFormat.Value(_metric, result.ValidationMetric);
                SizeF size = g.MeasureString(text, UiTheme.MonoFont);
                g.DrawString(text, UiTheme.MonoFont, textBrush, x + (w - size.Width) / 2, y + (h - size.Height) / 2);

                if (_best is { } best && best.HiddenLayers == result.HiddenLayers && best.Neurons == result.Neurons)
                {
                    g.DrawRectangle(frame, x + 1, y + 1, w - 2, h - 2);
                }
            }
        }
    }

    private static void DrawHeaders(Graphics g, Rectangle plot, SweepGrid grid, float cellWidth, float cellHeight)
    {
        using var headerBrush = new SolidBrush(UiTheme.TextMuted);
        for (int c = 0; c < grid.NeuronOptions.Length; c++)
        {
            string text = grid.NeuronOptions[c].ToString();
            SizeF size = g.MeasureString(text, UiTheme.UiFont);
            g.DrawString(text, UiTheme.UiFont, headerBrush, plot.Left + c * cellWidth + (cellWidth - size.Width) / 2, plot.Top - 18);
        }

        for (int r = 0; r < grid.LayerOptions.Length; r++)
        {
            string text = $"слоёв: {grid.LayerOptions[r]}";
            SizeF size = g.MeasureString(text, UiTheme.UiFont);
            g.DrawString(text, UiTheme.UiFont, headerBrush, plot.Left - size.Width - 8, plot.Top + r * cellHeight + (cellHeight - size.Height) / 2);
        }
    }

    /// <summary>Худший результат — красный, средний — золотой, лучший — зелёный.</summary>
    private Color ScaleColor(float value)
    {
        float range = _bestValue - _worstValue;
        float t = MathF.Abs(range) < 1e-9f ? 1f : Math.Clamp((value - _worstValue) / range, 0f, 1f);
        return t < 0.5f
            ? UiTheme.Mix(UiTheme.AccentRed, UiTheme.AccentGold, t * 2f)
            : UiTheme.Mix(UiTheme.AccentGold, UiTheme.AccentGreen, (t - 0.5f) * 2f);
    }
}
