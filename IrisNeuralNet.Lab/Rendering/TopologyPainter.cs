using System;
using System.Drawing;
using IrisNeuralNet.Core;
using IrisNeuralNet.Lab.Training;

namespace IrisNeuralNet.Lab.Rendering;

/// <summary>
/// Топология сети: рёбра — веса (цвет — знак, толщина и прозрачность — |w|), нейроны — активации
/// пробного образца. Широкие слои (MNIST: 784 входа, 512 нейронов) прореживаются до
/// <see cref="MaxNeuronsPerColumn"/> равномерно выбранных нейронов.
/// </summary>
internal static class TopologyPainter
{
    private const int MaxNeuronsPerColumn = 24;
    private const float WeakEdge = 0.04f;

    public static void Draw(SvgBuilder svg, RectangleF bounds, NeuralNetwork? network, ReadOnlySpan<float> probeSample)
    {
        svg.Rect(bounds, Theme.ChartBackground);
        if (network is null || probeSample.Length != network.InputDim)
        {
            svg.Text(10f, 10f, "ожидание сети... (нажми Start)", Theme.TextMuted);
            return;
        }

        // Своя проба на каждый кадр: поток обучения и параллельные запросы её не делят.
        var probe = new NetworkProbe();
        probe.Run(network, probeSample);

        svg.Text(8f, 6f, "Топология: веса и активации (проба — первый образец валидации)", Theme.TextPrimary, TextRole.Title);

        int columns = network.LayerCount + 1;
        RectangleF area = RectangleF.FromLTRB(16f, 30f, bounds.Right - 16f, bounds.Bottom - 48f);
        if (area.Width < 40f || area.Height < 40f) return;

        var counts = new int[columns];
        counts[0] = network.InputDim;
        for (int i = 0; i < network.LayerCount; i++) counts[i + 1] = network.Layers[i].OutputDim;

        var shown = new int[columns][];
        for (int c = 0; c < columns; c++) shown[c] = SampleNeurons(counts[c]);

        var columnX = new float[columns];
        for (int c = 0; c < columns; c++) columnX[c] = area.Left + c * (area.Width - 1f) / (columns - 1);

        float radius = 9f;
        foreach (int[] column in shown) radius = MathF.Min(radius, area.Height / (2.4f * column.Length));
        radius = MathF.Max(radius, 2.5f);

        for (int i = 0; i < network.LayerCount; i++)
        {
            DrawEdges(svg, network.Layers[i], shown[i], shown[i + 1], columnX[i], columnX[i + 1], area);
        }

        for (int c = 0; c < columns; c++)
        {
            for (int n = 0; n < shown[c].Length; n++)
            {
                int neuron = shown[c][n];
                float value = c == 0 ? probeSample[neuron] : probe.Activations[c - 1][neuron];
                float t = (value / (1f + MathF.Abs(value)) + 1f) / 2f;
                PointF p = Position(columnX[c], area, n, shown[c].Length);
                svg.Circle(p.X, p.Y, radius, Theme.Mix(Theme.PanelBackground, Theme.AccentGreen, t), c == 0 ? Theme.TextMuted : Theme.Border);
            }
        }

        // Подпись колонки в две строки: «L1 Relu 512» и, если слой прорежен, «показано 24» — узкие колонки не наезжают.
        for (int c = 0; c < columns; c++)
        {
            string label = c == 0 ? $"вход {counts[0]}" : $"L{c} {network.Layers[c - 1].ActivationKind} {counts[c]}";
            string sampled = shown[c].Length < counts[c] ? $"показано {shown[c].Length}" : string.Empty;
            float half = MathF.Max(SvgBuilder.EstimateWidth(label), SvgBuilder.EstimateWidth(sampled)) / 2f;
            float x = Math.Clamp(columnX[c], half, MathF.Max(half, bounds.Right - half));
            svg.Text(x, area.Bottom + 8f, label, Theme.TextMuted, anchor: TextAnchor.Middle);
            if (sampled.Length > 0) svg.Text(x, area.Bottom + 23f, sampled, Theme.TextMuted, anchor: TextAnchor.Middle);
        }
    }

    private static void DrawEdges(SvgBuilder svg, DenseLayer layer, int[] inputs, int[] outputs, float fromX, float toX, RectangleF area)
    {
        // Веса параллельно меняет поток обучения: читаем «как есть», зажимаем t и отсекаем NaN.
        ReadOnlySpan<float> weights = layer.Weights;
        float maxAbs = 1e-6f;
        foreach (int input in inputs)
        {
            foreach (int output in outputs) maxAbs = MathF.Max(maxAbs, MathF.Abs(weights[input * layer.OutputDim + output]));
        }

        for (int a = 0; a < inputs.Length; a++)
        {
            for (int b = 0; b < outputs.Length; b++)
            {
                float weight = weights[inputs[a] * layer.OutputDim + outputs[b]];
                float t = MathF.Min(1f, MathF.Abs(weight) / maxAbs);
                if (float.IsNaN(t) || t < WeakEdge) continue;

                Color color = weight >= 0f ? Theme.AccentBlue : Theme.AccentRed;
                PointF from = Position(fromX, area, a, inputs.Length);
                PointF to = Position(toX, area, b, outputs.Length);
                svg.Line(from.X, from.Y, to.X, to.Y, Color.FromArgb((int)(30 + 150 * t), color), 0.6f + 2.2f * t);
            }
        }
    }

    /// <summary>Индексы нейронов для отображения: все или равномерная выборка с первым и последним.</summary>
    internal static int[] SampleNeurons(int count)
    {
        int shown = Math.Min(count, MaxNeuronsPerColumn);
        var indices = new int[shown];
        for (int k = 0; k < shown; k++)
        {
            indices[k] = shown == 1 ? 0 : (int)Math.Round(k * (count - 1) / (double)(shown - 1));
        }

        return indices;
    }

    private static PointF Position(float x, RectangleF area, int index, int count) =>
        new(x, area.Top + (count == 1 ? area.Height / 2f : index * (area.Height - 1f) / (count - 1)));
}
