using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.Text;
using System.Text.Json;

namespace IrisNeuralNet.Lab.Rendering;

internal readonly record struct ChartSeries(string Name, Color Color, IReadOnlyList<float> Values);

/// <summary>
/// Линейный график по эпохам: область графика слева, легенда — в отдельной колонке справа
/// (текст и кривые не пересекаются), в легенде — последнее значение ряда.
/// Все ряды делят одну ось эпох, ряды из NaN (обучение без валидации) пропускаются.
/// Данные рядов кладутся в data-hover корня — по ним страница рисует подсказку при наведении.
/// </summary>
internal static class ChartPainter
{
    private const float LeftMargin = 62f;
    private const float TopMargin = 30f;
    private const float MinLegendWidth = 150f;
    private const float BottomMargin = 26f;
    private const int GridRows = 4;

    /// <param name="withHover">Данные для подсказки при наведении — нужны странице, но не файлам отчётов.</param>
    public static void Draw(
        SvgBuilder svg, RectangleF bounds, string title, IReadOnlyList<ChartSeries> series, ChartStyle style, bool withHover = true)
    {
        svg.Rect(bounds, style.Background);

        RectangleF plot = RectangleF.FromLTRB(
            bounds.Left + LeftMargin,
            bounds.Top + TopMargin,
            bounds.Right - LegendWidth(series, bounds),
            bounds.Bottom - BottomMargin);
        if (plot.Width < 10f || plot.Height < 10f) return;

        svg.Text(bounds.Left + 8f, bounds.Top + 6f, title, style.Text, TextRole.Title);
        svg.Rect(plot, fill: null, stroke: style.Border);

        if (!TryGetRange(series, out float min, out float max))
        {
            svg.Text(plot.Left + 8f, plot.Top + 8f, "ожидание данных...", style.MutedText);
            DrawLegend(svg, plot, series, style);
            return;
        }

        int epochCount = MaxCount(series);
        DrawGrid(svg, bounds, plot, min, max, style);
        foreach (ChartSeries s in series)
        {
            DrawSeries(svg, plot, s, epochCount, min, max);
        }

        DrawLegend(svg, plot, series, style);
        svg.Text(plot.Left, plot.Bottom + 7f, $"эпох: {epochCount}", style.MutedText, TextRole.Mono);
        if (withHover) svg.Data("hover", HoverData(plot, series, epochCount, min, max));
    }

    private static bool TryGetRange(IReadOnlyList<ChartSeries> series, out float min, out float max)
    {
        min = float.MaxValue;
        max = float.MinValue;
        foreach (ChartSeries s in series)
        {
            foreach (float v in s.Values)
            {
                if (!float.IsFinite(v)) continue;
                min = MathF.Min(min, v);
                max = MathF.Max(max, v);
            }
        }

        if (min > max) return false;
        if (max - min < 1e-9f) max = min + 1f;
        return true;
    }

    private static void DrawGrid(SvgBuilder svg, RectangleF bounds, RectangleF plot, float min, float max, ChartStyle style)
    {
        for (int i = 0; i <= GridRows; i++)
        {
            float y = plot.Top + i * (plot.Height - 1f) / GridRows;
            float value = max - i * (max - min) / GridRows;
            if (i > 0 && i < GridRows)
            {
                svg.Line(plot.Left + 1f, y, plot.Right - 1f, y, style.Grid, dotted: true);
            }

            svg.Text(bounds.Left + 6f, y - 7f, value.ToString("F3", CultureInfo.InvariantCulture), style.MutedText, TextRole.Mono);
        }
    }

    /// <param name="epochCount">Длина самого длинного ряда: ряд из 150 эпох не растягивается до 500.</param>
    private static void DrawSeries(SvgBuilder svg, RectangleF plot, ChartSeries series, int epochCount, float min, float max)
    {
        IReadOnlyList<float> values = series.Values;
        if (values.Count < 2 || !AllFinite(values)) return;

        var points = new List<PointF>(Math.Min(values.Count, (int)plot.Width * 2 + 2));
        int bucketSize = Math.Max(1, (int)MathF.Ceiling(values.Count / (plot.Width * 2f)));
        for (int start = 0; start < values.Count; start += bucketSize)
        {
            AddBucket(points, plot, values, start, Math.Min(values.Count, start + bucketSize), epochCount, min, max);
        }

        PointF last = MapPoint(plot, values.Count - 1, epochCount, values[^1], min, max);
        if (points[^1] != last) points.Add(last);

        svg.Polyline(points, series.Color, 2f);
        svg.Circle(last.X, last.Y, 3f, series.Color);
    }

    /// <summary>
    /// Прореживание длинных рядов (5000 эпох на 700 пикселей): из каждой группы берутся
    /// первая точка, минимум и максимум — форма кривой и выбросы сохраняются.
    /// </summary>
    private static void AddBucket(
        List<PointF> points, RectangleF plot, IReadOnlyList<float> values, int start, int end, int epochCount, float min, float max)
    {
        if (end - start <= 2)
        {
            for (int i = start; i < end; i++) points.Add(MapPoint(plot, i, epochCount, values[i], min, max));
            return;
        }

        int low = start, high = start;
        for (int i = start + 1; i < end; i++)
        {
            if (values[i] < values[low]) low = i;
            if (values[i] > values[high]) high = i;
        }

        points.Add(MapPoint(plot, start, epochCount, values[start], min, max));
        int first = Math.Min(low, high), second = Math.Max(low, high);
        if (first != start) points.Add(MapPoint(plot, first, epochCount, values[first], min, max));
        if (second != first) points.Add(MapPoint(plot, second, epochCount, values[second], min, max));
    }

    /// <summary>Легенда в колонке справа: имя ряда (его цветом) и последнее значение.</summary>
    private static void DrawLegend(SvgBuilder svg, RectangleF plot, IReadOnlyList<ChartSeries> series, ChartStyle style)
    {
        float x = plot.Right + 12f;
        float y = plot.Top + 2f;
        foreach (ChartSeries s in series)
        {
            svg.Line(x, y + 7f, x + 16f, y + 7f, s.Color, 2f);
            svg.Text(x + 21f, y, s.Name, s.Color);

            if (s.Values.Count > 0 && float.IsFinite(s.Values[^1]))
            {
                svg.Text(x + 21f, y + 15f, s.Values[^1].ToString("F4", CultureInfo.InvariantCulture), style.Text, TextRole.Mono);
                y += 36f;
            }
            else
            {
                y += 20f;
            }
        }
    }

    /// <summary>JSON для подсказки: прямоугольник графика, число эпох, диапазон и ряды.</summary>
    private static string HoverData(RectangleF plot, IReadOnlyList<ChartSeries> series, int epochCount, float min, float max)
    {
        var json = new StringBuilder();
        json.Append("{\"plot\":[").Append(SvgBuilder.F(plot.Left)).Append(',').Append(SvgBuilder.F(plot.Top)).Append(',')
            .Append(SvgBuilder.F(plot.Width)).Append(',').Append(SvgBuilder.F(plot.Height))
            .Append("],\"count\":").Append(epochCount)
            .Append(",\"min\":").Append(Number(min)).Append(",\"max\":").Append(Number(max))
            .Append(",\"series\":[");
        for (int s = 0; s < series.Count; s++)
        {
            if (s > 0) json.Append(',');
            json.Append("{\"name\":").Append(JsonSerializer.Serialize(series[s].Name))
                .Append(",\"color\":\"").Append(Theme.Hex(series[s].Color)).Append("\",\"values\":[");
            IReadOnlyList<float> values = series[s].Values;
            for (int i = 0; i < values.Count; i++)
            {
                if (i > 0) json.Append(',');
                json.Append(Number(values[i]));
            }

            json.Append("]}");
        }

        return json.Append("]}").ToString();
    }

    private static string Number(float value) =>
        float.IsFinite(value) ? value.ToString("G5", CultureInfo.InvariantCulture) : "null";

    private static PointF MapPoint(RectangleF plot, int index, int count, float value, float min, float max) =>
        new(
            plot.Left + index * (plot.Width - 1f) / Math.Max(1, count - 1),
            plot.Bottom - 1f - (value - min) / (max - min) * (plot.Height - 1f));

    private static bool AllFinite(IReadOnlyList<float> values)
    {
        for (int i = 0; i < values.Count; i++)
        {
            if (!float.IsFinite(values[i])) return false;
        }

        return true;
    }

    /// <summary>Ширина колонки легенды по самому длинному имени ряда (не больше половины графика).</summary>
    private static float LegendWidth(IReadOnlyList<ChartSeries> series, RectangleF bounds)
    {
        float widest = 0f;
        foreach (ChartSeries s in series) widest = MathF.Max(widest, SvgBuilder.EstimateWidth(s.Name));
        return Math.Clamp(widest + 40f, MinLegendWidth, MathF.Max(MinLegendWidth, bounds.Width / 2f));
    }

    private static int MaxCount(IReadOnlyList<ChartSeries> series)
    {
        int count = 0;
        foreach (ChartSeries s in series) count = Math.Max(count, s.Values.Count);
        return count;
    }
}
