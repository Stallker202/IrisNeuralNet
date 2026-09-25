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

/// <summary>
/// Векторный рендерер графиков: область графика слева, легенда — в отдельной колонке справа,
/// поэтому текст и кривые не пересекаются. Легенда показывает последнее значение каждой серии.
/// </summary>
internal static class ChartPainter
{
    private const int LeftMargin = 62;
    private const int TopMargin = 30;
    private const int RightMargin = 150;
    private const int BottomMargin = 26;
    private const int GridRows = 4;

    public static void Draw(Graphics g, Rectangle bounds, string title, IReadOnlyList<ChartSeries> series)
    {
        g.SmoothingMode = SmoothingMode.AntiAlias;

        using (var background = new SolidBrush(UiTheme.ChartBackground))
        {
            g.FillRectangle(background, bounds);
        }

        Rectangle plot = Rectangle.FromLTRB(
            bounds.Left + LeftMargin,
            bounds.Top + TopMargin,
            bounds.Right - RightMargin,
            bounds.Bottom - BottomMargin);
        if (plot.Width < 10 || plot.Height < 10)
        {
            return;
        }

        DrawTitle(g, bounds, title);
        DrawFrame(g, plot);

        float min = float.MaxValue;
        float max = float.MinValue;
        bool any = false;
        foreach (ChartSeries s in series)
        {
            foreach (float v in s.Values)
            {
                min = MathF.Min(min, v);
                max = MathF.Max(max, v);
                any = true;
            }
        }

        if (!any)
        {
            using var waiting = new SolidBrush(UiTheme.TextMuted);
            g.DrawString("ожидание данных...", UiTheme.UiFont, waiting, plot.Left + 8, plot.Top + 8);
            DrawLegend(g, plot, series);
            return;
        }

        if (max - min < 1e-9f)
        {
            max = min + 1f;
        }

        DrawGrid(g, bounds, plot, min, max);

        foreach (ChartSeries s in series)
        {
            DrawSeries(g, plot, s, min, max);
        }

        DrawLegend(g, plot, series);

        using var epochs = new SolidBrush(UiTheme.TextMuted);
        g.DrawString($"epochs: {MaxCount(series)}", UiTheme.MonoFont, epochs, plot.Left, plot.Bottom + 7);
    }

    private static void DrawTitle(Graphics g, Rectangle bounds, string title)
    {
        using var brush = new SolidBrush(UiTheme.TextPrimary);
        g.DrawString(title, UiTheme.TitleFont, brush, bounds.Left + 8, bounds.Top + 6);
    }

    private static void DrawFrame(Graphics g, Rectangle plot)
    {
        using var pen = new Pen(UiTheme.Border);
        g.DrawRectangle(pen, plot);
    }

    private static void DrawGrid(Graphics g, Rectangle bounds, Rectangle plot, float min, float max)
    {
        using var gridPen = new Pen(UiTheme.GridLine) { DashStyle = DashStyle.Dot };
        using var labelBrush = new SolidBrush(UiTheme.TextMuted);
        for (int i = 0; i <= GridRows; i++)
        {
            float y = plot.Top + i * (plot.Height - 1) / (float)GridRows;
            float value = max - i * (max - min) / GridRows;
            if (i > 0 && i < GridRows)
            {
                g.DrawLine(gridPen, plot.Left + 1, y, plot.Right - 1, y);
            }

            g.DrawString(value.ToString("F3", CultureInfo.InvariantCulture), UiTheme.MonoFont, labelBrush,
                bounds.Left + 6, y - 7);
        }
    }

    private static void DrawSeries(Graphics g, Rectangle plot, ChartSeries series, float min, float max)
    {
        if (series.Values.Count < 2)
        {
            return;
        }

        var points = new PointF[series.Values.Count];
        for (int i = 0; i < series.Values.Count; i++)
        {
            points[i] = MapPoint(plot, i, series.Values.Count, series.Values[i], min, max);
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
    private static void DrawLegend(Graphics g, Rectangle plot, IReadOnlyList<ChartSeries> series)
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
                g.DrawString(s.Name, UiTheme.UiFont, nameBrush, x + 21, y);
            }

            if (s.Values.Count > 0)
            {
                using var valueBrush = new SolidBrush(UiTheme.TextPrimary);
                string value = s.Values[s.Values.Count - 1].ToString("F4", CultureInfo.InvariantCulture);
                g.DrawString(value, UiTheme.MonoFont, valueBrush, x + 21, y + 15);
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