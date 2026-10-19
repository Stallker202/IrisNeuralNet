using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Text;
using IrisNeuralNet.Diagnostics;
using IrisNeuralNet.Lab.Imaging;
using IrisNeuralNet.Lab.Reporting;

namespace IrisNeuralNet.Lab.Rendering;

/// <summary>
/// Графики отчётов в SVG тем же <see cref="ChartPainter"/>, что и на странице лаборатории,
/// но в светлой теме. Работает на любой ОС без зависимостей; GitHub и Word показывают SVG,
/// для PNG — кнопка у графиков в браузере.
/// </summary>
internal sealed class SvgChartRenderer : IChartRenderer
{
    private const float PanelWidth = 640f;
    private const float PanelHeight = 400f;
    private const float WideExtra = 260f;
    private const int DigitScale = 4;
    private const int DigitsPerRow = 8;
    private const float CaptionHeight = 20f;
    private const float TitleHeight = 30f;

    /// <summary>Tableau 10: различимые на белом фоне цвета рядов.</summary>
    private static readonly Color[] Palette =
    {
        Color.FromArgb(78, 121, 167), Color.FromArgb(242, 142, 43), Color.FromArgb(89, 161, 79), Color.FromArgb(225, 87, 89),
        Color.FromArgb(176, 122, 161), Color.FromArgb(156, 117, 95), Color.FromArgb(237, 201, 72), Color.FromArgb(118, 183, 178),
        Color.FromArgb(255, 157, 167), Color.FromArgb(186, 176, 172),
    };

    public string? Extension => ".svg";

    public void SaveHistory(string path, string title, TrainingHistory history, string metricLabel)
    {
        var svg = new SvgBuilder(PanelWidth * 2f, PanelHeight);
        ChartPainter.Draw(svg, new RectangleF(0f, 0f, PanelWidth, PanelHeight), $"{title} — потери", new[]
        {
            new ChartSeries("обучение", Palette[0], history.Series(m => m.TrainLoss)),
            new ChartSeries("валидация", Palette[1], history.Series(m => m.ValidationLoss)),
        }, ChartStyle.Light, withHover: false);
        ChartPainter.Draw(svg, new RectangleF(PanelWidth, 0f, PanelWidth, PanelHeight), $"{title} — {metricLabel}", new[]
        {
            new ChartSeries("обучение", Palette[0], history.Series(m => m.TrainMetric)),
            new ChartSeries("валидация", Palette[1], history.Series(m => m.ValidationMetric)),
        }, ChartStyle.Light, withHover: false);
        Save(path, svg);
    }

    public void SaveLines(string path, string title, string yLabel, IReadOnlyList<PlotSeries> series)
    {
        var colored = new ChartSeries[series.Count];
        for (int i = 0; i < series.Count; i++)
        {
            colored[i] = new ChartSeries(series[i].Name, Palette[i % Palette.Length], series[i].Values);
        }

        // Подпись оси Y — в заголовке, если её там ещё нет; легенде с длинными именами нужна ширина.
        string caption = title.Contains(yLabel, StringComparison.OrdinalIgnoreCase) ? title : $"{title} ({yLabel})";
        var svg = new SvgBuilder(PanelWidth + WideExtra, PanelHeight);
        ChartPainter.Draw(svg, new RectangleF(0f, 0f, PanelWidth + WideExtra, PanelHeight), caption, colored, ChartStyle.Light, withHover: false);
        Save(path, svg);
    }

    public void SaveDigits(string path, string title, IReadOnlyList<DigitSample> samples)
    {
        if (samples.Count == 0) return;

        float cell = 28 * DigitScale;
        int columns = Math.Min(DigitsPerRow, samples.Count);
        int rows = (samples.Count + columns - 1) / columns;
        var svg = new SvgBuilder(columns * cell, TitleHeight + rows * (cell + CaptionHeight));
        ChartStyle style = ChartStyle.Light;
        svg.Rect(0f, 0f, svg.Width, svg.Height, style.Background);
        svg.Text(6f, 6f, title, style.Text, TextRole.Title);
        for (int i = 0; i < samples.Count; i++)
        {
            float x = i % columns * cell;
            float y = TitleHeight + i / columns * (cell + CaptionHeight);
            svg.Image(x, y, cell, cell, PngCodec.ToDataUri(DigitPng.Encode(samples[i].Pixels, inverted: true)), pixelated: true);
            svg.Text(x + cell / 2f, y + cell + 2f, samples[i].Caption, style.Text, anchor: TextAnchor.Middle);
        }

        Save(path, svg);
    }

    private static void Save(string path, SvgBuilder svg) =>
        File.WriteAllText(path, svg.ToString(), new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
}

/// <summary>Изображение MNIST 28×28 (пиксели 0..255) → PNG.</summary>
internal static class DigitPng
{
    /// <param name="inverted">Чёрные чернила на белом (как plt.imshow с cmap=binary) вместо белых на чёрном.</param>
    public static byte[] Encode(ReadOnlySpan<float> pixels, bool inverted)
    {
        int side = (int)Math.Sqrt(pixels.Length);
        if (side * side != pixels.Length) throw new ArgumentException("Digit image must be square.", nameof(pixels));

        return PngCodec.EncodeGray(ToBytes(pixels, inverted), side, side);
    }

    /// <summary>Пиксели 0..255 → байты (с отсечением и защитой от NaN).</summary>
    public static byte[] ToBytes(ReadOnlySpan<float> pixels, bool inverted)
    {
        var bytes = new byte[pixels.Length];
        for (int i = 0; i < bytes.Length; i++)
        {
            int value = (int)Math.Clamp(float.IsNaN(pixels[i]) ? 0f : pixels[i], 0f, 255f);
            bytes[i] = (byte)(inverted ? 255 - value : value);
        }

        return bytes;
    }
}
