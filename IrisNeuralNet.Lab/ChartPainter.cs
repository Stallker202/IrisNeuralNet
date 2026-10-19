using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Globalization;

namespace IrisNeuralNet.Lab;

internal readonly struct ChartSeries
{
    public ChartSeries(string name, Color color, IReadOnlyList<float> values)
    {
        Name = name;
        Color = color;
        Values = values;
    }

    public string Name { get; }
    public Color Color { get; }
    public IReadOnlyList<float> Values { get; }
}

/// <summary>Палитра и шрифты графика: тёмная — для окна лаборатории, светлая — для PNG в отчёт.</summary>
internal sealed record ChartStyle(
    Color Background, Color Text, Color MutedText, Color Grid, Color Border, Font TitleFont, Font LabelFont, Font MonoFont)
{
    public static readonly ChartStyle Dark = new(
        UiTheme.ChartBackground, UiTheme.TextPrimary, UiTheme.TextMuted, UiTheme.GridLine, UiTheme.Border,
        UiTheme.TitleFont, UiTheme.UiFont, UiTheme.MonoFont);

    public static readonly ChartStyle Light = new(
        Color.White, Color.FromArgb(33, 37, 41), Color.FromArgb(96, 103, 112), Color.FromArgb(226, 229, 233), Color.FromArgb(173, 181, 189),
        new Font("Segoe UI", 10f, FontStyle.Bold), new Font("Segoe UI", 9f), new Font("Consolas", 9f));
}

/// <summary>
/// Векторный рендерер графиков: область графика слева, легенда — в отдельной колонке справа,
/// поэтому текст и кривые не пересекаются. Легенда показывает последнее значение каждой серии.
/// Ряды целиком из NaN (обучение без валидации) пропускаются.
/// </summary>
internal static class ChartPainter
{
    private const int LeftMargin = 62;
    private const int TopMargin = 30;
    private const int MinLegendWidth = 150;
    private const int BottomMargin = 26;
    private const int GridRows = 4;

    public static void Draw(Graphics g, Rectangle bounds, string title, IReadOnlyList<ChartSeries> series, ChartStyle? style = null)
    {
        style ??= ChartStyle.Dark;
        g.SmoothingMode = SmoothingMode.AntiAlias;

        using (var background = new SolidBrush(style.Background))
        {
            g.FillRectangle(background, bounds);
        }

        Rectangle plot = Rectangle.FromLTRB(
            bounds.Left + LeftMargin,
            bounds.Top + TopMargin,
            bounds.Right - LegendWidth(g, series, style, bounds),
            bounds.Bottom - BottomMargin);
        if (plot.Width < 10 || plot.Height < 10)
        {
            return;
        }

        DrawTitle(g, bounds, title, style);
        DrawFrame(g, plot, style);

        float min = float.MaxValue;
        float max = float.MinValue;
        bool any = false;
        foreach (ChartSeries s in series)
        {
            foreach (float v in s.Values)
            {
                if (!float.IsFinite(v)) continue;
                min = MathF.Min(min, v);
                max = MathF.Max(max, v);
                any = true;
            }
        }

        if (!any)
        {
            using var waiting = new SolidBrush(style.MutedText);
            g.DrawString("ожидание данных...", style.LabelFont, waiting, plot.Left + 8, plot.Top + 8);
            DrawLegend(g, plot, series, style);
            return;
        }

        if (max - min < 1e-9f)
        {
            max = min + 1f;
        }

        DrawGrid(g, bounds, plot, min, max, style);

        foreach (ChartSeries s in series)
        {
            DrawSeries(g, plot, s, MaxCount(series), min, max);
        }

        DrawLegend(g, plot, series, style);

        using var epochs = new SolidBrush(style.MutedText);
        g.DrawString($"эпох: {MaxCount(series)}", style.MonoFont, epochs, plot.Left, plot.Bottom + 7);
    }

    private static void DrawTitle(Graphics g, Rectangle bounds, string title, ChartStyle style)
    {
        using var brush = new SolidBrush(style.Text);
        g.DrawString(title, style.TitleFont, brush, bounds.Left + 8, bounds.Top + 6);
    }

    private static void DrawFrame(Graphics g, Rectangle plot, ChartStyle style)
    {
        using var pen = new Pen(style.Border);
        g.DrawRectangle(pen, plot);
    }

    private static void DrawGrid(Graphics g, Rectangle bounds, Rectangle plot, float min, float max, ChartStyle style)
    {
        using var gridPen = new Pen(style.Grid) { DashStyle = DashStyle.Dot };
        using var labelBrush = new SolidBrush(style.MutedText);
        for (int i = 0; i <= GridRows; i++)
        {
            float y = plot.Top + i * (plot.Height - 1) / (float)GridRows;
            float value = max - i * (max - min) / GridRows;
            if (i > 0 && i < GridRows)
            {
                g.DrawLine(gridPen, plot.Left + 1, y, plot.Right - 1, y);
            }

            g.DrawString(value.ToString("F3", CultureInfo.InvariantCulture), style.MonoFont, labelBrush,
                bounds.Left + 6, y - 7);
        }
    }

    /// <param name="epochCount">Длина самого длинного ряда: ось X общая, ряд из 150 эпох не растягивается до 500.</param>
    private static void DrawSeries(Graphics g, Rectangle plot, ChartSeries series, int epochCount, float min, float max)
    {
        if (series.Values.Count < 2 || !AllFinite(series.Values))
        {
            return;
        }

        var points = new PointF[series.Values.Count];
        for (int i = 0; i < series.Values.Count; i++)
        {
            points[i] = MapPoint(plot, i, epochCount, series.Values[i], min, max);
        }

        using (var pen = new Pen(series.Color, 2f))
        {
            g.DrawLines(pen, points);
        }

        PointF last = points[points.Length - 1];
        using var marker = new SolidBrush(series.Color);
        g.FillEllipse(marker, last.X - 3, last.Y - 3, 6, 6);
    }

    /// <summary>Легенда в отдельной колонке справа от графика: имя серии и её последнее значение.</summary>
    private static void DrawLegend(Graphics g, Rectangle plot, IReadOnlyList<ChartSeries> series, ChartStyle style)
    {
        float x = plot.Right + 12;
        float y = plot.Top + 2;
        foreach (ChartSeries s in series)
        {
            using (var pen = new Pen(s.Color, 2f))
            {
                g.DrawLine(pen, x, y + 7, x + 16, y + 7);
            }

            using (var nameBrush = new SolidBrush(s.Color))
            {
                g.DrawString(s.Name, style.LabelFont, nameBrush, x + 21, y);
            }

            if (s.Values.Count > 0 && float.IsFinite(s.Values[s.Values.Count - 1]))
            {
                using var valueBrush = new SolidBrush(style.Text);
                string value = s.Values[s.Values.Count - 1].ToString("F4", CultureInfo.InvariantCulture);
                g.DrawString(value, style.MonoFont, valueBrush, x + 21, y + 15);
                y += 36;
            }
            else
            {
                y += 20;
            }
        }
    }

    private static PointF MapPoint(Rectangle plot, int index, int count, float value, float min, float max)
    {
        float x = plot.Left + index * (plot.Width - 1) / (float)(count - 1);
        float y = plot.Bottom - 1 - (value - min) / (max - min) * (plot.Height - 1);
        return new PointF(x, y);
    }

    private static bool AllFinite(IReadOnlyList<float> values)
    {
        for (int i = 0; i < values.Count; i++)
        {
            if (!float.IsFinite(values[i])) return false;
        }

        return true;
    }

    /// <summary>Ширина колонки легенды по самому длинному имени ряда (не больше половины графика).</summary>
    private static int LegendWidth(Graphics g, IReadOnlyList<ChartSeries> series, ChartStyle style, Rectangle bounds)
    {
        float widest = 0f;
        foreach (ChartSeries s in series)
        {
            widest = MathF.Max(widest, g.MeasureString(s.Name, style.LabelFont).Width);
        }

        return Math.Clamp((int)widest + 40, MinLegendWidth, Math.Max(MinLegendWidth, bounds.Width / 2));
    }

    private static int MaxCount(IReadOnlyList<ChartSeries> series)
    {
        int count = 0;
        foreach (ChartSeries s in series)
        {
            count = Math.Max(count, s.Values.Count);
        }

        return count;
    }
}
