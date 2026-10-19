using System;
using System.Drawing;
using System.Globalization;
using IrisNeuralNet.Core;

namespace IrisNeuralNet.Lab.Rendering;

/// <summary>Горизонтальные столбики вероятностей классов (softmax); максимум подсвечен.</summary>
internal static class ProbabilityBarsPainter
{
    public static void Draw(SvgBuilder svg, RectangleF bounds, ReadOnlySpan<float> probabilities)
    {
        svg.Rect(bounds, Theme.ChartBackground);
        svg.Text(8f, 6f, "Вероятности классов (softmax)", Theme.TextPrimary, TextRole.Title);
        if (probabilities.IsEmpty) return;

        int best = Metrics.ArgMax(probabilities);
        float rowHeight = (bounds.Height - 40f) / probabilities.Length;
        const float barLeft = 40f;
        float barMax = MathF.Max(10f, bounds.Width - barLeft - 80f);
        for (int digit = 0; digit < probabilities.Length; digit++)
        {
            float y = 34f + digit * rowHeight;
            float p = Math.Clamp(float.IsNaN(probabilities[digit]) ? 0f : probabilities[digit], 0f, 1f);
            svg.Text(14f, y + rowHeight / 2f - 8f, digit.ToString(CultureInfo.InvariantCulture), Theme.TextMuted, TextRole.Title);
            svg.Rect(barLeft, y + rowHeight * 0.2f, MathF.Max(1f, p * barMax), rowHeight * 0.6f, digit == best ? Theme.AccentGreen : Theme.AccentBlue);
            svg.Text(barLeft + p * barMax + 6f, y + rowHeight / 2f - 7f, p.ToString("P1", CultureInfo.InvariantCulture), Theme.TextMuted, TextRole.Mono);
        }
    }
}
