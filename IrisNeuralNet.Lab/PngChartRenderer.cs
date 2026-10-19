using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Drawing.Text;
using IrisNeuralNet.Diagnostics;
using IrisNeuralNet.Lab.Reporting;

namespace IrisNeuralNet.Lab;

/// <summary>PNG-графики для отчёта тем же <see cref="ChartPainter"/>, что и в окне, но в светлой теме.</summary>
internal sealed class PngChartRenderer : IChartRenderer
{
    private const int PanelWidth = 640;
    private const int PanelHeight = 400;
    private const int DigitScale = 4;
    private const int DigitsPerRow = 8;
    private const int CaptionHeight = 20;

    /// <summary>Tableau 10: различимые на белом фоне цвета для рядов.</summary>
    private static readonly Color[] Palette =
    {
        Color.FromArgb(78, 121, 167), Color.FromArgb(242, 142, 43), Color.FromArgb(89, 161, 79), Color.FromArgb(225, 87, 89),
        Color.FromArgb(176, 122, 161), Color.FromArgb(156, 117, 95), Color.FromArgb(237, 201, 72), Color.FromArgb(118, 183, 178),
        Color.FromArgb(255, 157, 167), Color.FromArgb(186, 176, 172),
    };

    public string? Extension => ".png";

    public void SaveHistory(string path, string title, TrainingHistory history, string metricLabel)
    {
        var loss = new[]
        {
            new ChartSeries("обучение", Palette[0], history.Series(m => m.TrainLoss)),
            new ChartSeries("валидация", Palette[1], history.Series(m => m.ValidationLoss)),
        };
        var metric = new[]
        {
            new ChartSeries("обучение", Palette[0], history.Series(m => m.TrainMetric)),
            new ChartSeries("валидация", Palette[1], history.Series(m => m.ValidationMetric)),
        };

        Save(path, PanelWidth * 2, PanelHeight, g =>
        {
            ChartPainter.Draw(g, new Rectangle(0, 0, PanelWidth, PanelHeight), $"{title} — потери", loss, ChartStyle.Light);
            ChartPainter.Draw(g, new Rectangle(PanelWidth, 0, PanelWidth, PanelHeight), $"{title} — {metricLabel}", metric, ChartStyle.Light);
        });
    }

    public void SaveLines(string path, string title, string yLabel, IReadOnlyList<PlotSeries> series)
    {
        var colored = new ChartSeries[series.Count];
        for (int i = 0; i < series.Count; i++)
        {
            colored[i] = new ChartSeries(series[i].Name, Palette[i % Palette.Length], series[i].Values);
        }

        // Легенда с длинными именами моделей требует ширины.
        Save(path, PanelWidth + 260, PanelHeight, g =>
            ChartPainter.Draw(g, new Rectangle(0, 0, PanelWidth + 260, PanelHeight), $"{title} ({yLabel})", colored, ChartStyle.Light));
    }

    public void SaveDigits(string path, string title, IReadOnlyList<DigitSample> samples)
    {
        if (samples.Count == 0) return;

        int cell = 28 * DigitScale;
        int columns = Math.Min(DigitsPerRow, samples.Count);
        int rows = (samples.Count + columns - 1) / columns;
        int titleHeight = 30;
        Save(path, columns * cell, titleHeight + rows * (cell + CaptionHeight), g =>
        {
            ChartStyle style = ChartStyle.Light;
            using var text = new SolidBrush(style.Text);
            g.DrawString(title, style.TitleFont, text, 6, 6);
            g.InterpolationMode = InterpolationMode.NearestNeighbor;
            g.PixelOffsetMode = PixelOffsetMode.Half;
            for (int i = 0; i < samples.Count; i++)
            {
                int x = i % columns * cell;
                int y = titleHeight + i / columns * (cell + CaptionHeight);
                using Bitmap digit = BitmapPixels.FromMnist(samples[i].Pixels, inverted: true);
                g.DrawImage(digit, new Rectangle(x, y, cell, cell));
                SizeF size = g.MeasureString(samples[i].Caption, style.LabelFont);
                g.DrawString(samples[i].Caption, style.LabelFont, text, x + (cell - size.Width) / 2, y + cell + 2);
            }
        });
    }

    private static void Save(string path, int width, int height, Action<Graphics> draw)
    {
        using var bitmap = new Bitmap(width, height);
        using (Graphics g = Graphics.FromImage(bitmap))
        {
            g.Clear(ChartStyle.Light.Background);
            g.TextRenderingHint = TextRenderingHint.AntiAliasGridFit;
            draw(g);
        }

        bitmap.Save(path, ImageFormat.Png);
    }
}
