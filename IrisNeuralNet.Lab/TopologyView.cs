using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;
using IrisNeuralNet.Core;

namespace IrisNeuralNet.Lab;

/// <summary>
/// Топология сети: рёбра — веса (цвет — знак, толщина и альфа — |w|), нейроны — активации пробы.
/// Широкие слои (MNIST: 784 входа, 512 нейронов) прореживаются до <see cref="MaxNeuronsPerColumn"/>
/// равномерно выбранных нейронов, иначе кадр рисовал бы сотни тысяч линий.
/// </summary>
internal sealed class TopologyView : PictureBox
{
    private const int MaxNeuronsPerColumn = 24;

    private readonly NetworkProbe _probe = new();
    private NeuralNetwork? _network;
    private float[] _probeSample = Array.Empty<float>();

    public TopologyView()
    {
        SetStyle(ControlStyles.OptimizedDoubleBuffer | ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint, true);
        UpdateStyles();
    }

    public void UpdateSource(NeuralNetwork? network, float[] probeSample)
    {
        _network = network;
        _probeSample = probeSample;
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

        NeuralNetwork? network = _network;
        if (network is null || _probeSample.Length != network.InputDim)
        {
            using var waiting = new SolidBrush(UiTheme.TextMuted);
            g.DrawString("ожидание сети... (нажми Start)", UiTheme.UiFont, waiting, 10, 10);
            return;
        }

        _probe.Run(network, _probeSample);

        using var title = new SolidBrush(UiTheme.TextPrimary);
        g.DrawString("Топология: веса и активации (проба — первый образец валидации)", UiTheme.TitleFont, title, 8, 6);

        int columns = network.LayerCount + 1;
        Rectangle area = Rectangle.FromLTRB(16, 30, ClientRectangle.Right - 16, ClientRectangle.Bottom - 34);
        if (area.Width < 40 || area.Height < 40) return;

        var shown = new int[columns][];
        var counts = new int[columns];
        counts[0] = network.InputDim;
        for (int i = 0; i < network.LayerCount; i++) counts[i + 1] = network.Layers[i].OutputDim;
        for (int c = 0; c < columns; c++) shown[c] = SampleNeurons(counts[c]);

        var columnX = new float[columns];
        for (int c = 0; c < columns; c++) columnX[c] = area.Left + c * (area.Width - 1) / (float)(columns - 1);

        float radius = 9f;
        foreach (int[] column in shown) radius = MathF.Min(radius, area.Height / (2.4f * column.Length));
        radius = MathF.Max(radius, 2.5f);

        for (int i = 0; i < network.LayerCount; i++)
        {
            DrawEdges(g, network.Layers[i], shown[i], shown[i + 1], columnX[i], columnX[i + 1], area);
        }

        for (int c = 0; c < columns; c++)
        {
            for (int n = 0; n < shown[c].Length; n++)
            {
                int neuron = shown[c][n];
                float value = c == 0 ? _probeSample[neuron] : _probe.Activations[c - 1][neuron];
                float t = (value / (1f + MathF.Abs(value)) + 1f) / 2f;
                PointF p = Position(columnX[c], area, n, shown[c].Length);
                using var fill = new SolidBrush(UiTheme.Mix(UiTheme.PanelBackground, UiTheme.AccentGreen, t));
                using var stroke = new Pen(c == 0 ? UiTheme.TextMuted : UiTheme.Border);
                g.FillEllipse(fill, p.X - radius, p.Y - radius, radius * 2, radius * 2);
                g.DrawEllipse(stroke, p.X - radius, p.Y - radius, radius * 2, radius * 2);
            }
        }

        using var muted = new SolidBrush(UiTheme.TextMuted);
        for (int c = 0; c < columns; c++)
        {
            string sampled = shown[c].Length < counts[c] ? $" (показано {shown[c].Length})" : string.Empty;
            string label = c == 0
                ? $"вход {counts[0]}{sampled}"
                : $"L{c} {network.Layers[c - 1].ActivationKind} {counts[c]}{sampled}";
            SizeF size = g.MeasureString(label, UiTheme.UiFont);
            float x = Math.Clamp(columnX[c] - size.Width / 2, 0, Math.Max(0, ClientRectangle.Right - size.Width));
            g.DrawString(label, UiTheme.UiFont, muted, x, area.Bottom + 8);
        }
    }

    private static void DrawEdges(Graphics g, DenseLayer layer, int[] inputs, int[] outputs, float fromX, float toX, Rectangle area)
    {
        // Веса параллельно пишет поток обучения: читаем «как есть», зажимаем t и отсекаем NaN,
        // иначе GDI+ упадёт на альфе или толщине пера.
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
                if (float.IsNaN(t) || t < 0.04f) continue;

                Color color = weight >= 0f ? UiTheme.AccentBlue : UiTheme.AccentRed;
                using var pen = new Pen(Color.FromArgb((int)(30 + 150 * t), color), 0.6f + 2.2f * t);
                g.DrawLine(pen, Position(fromX, area, a, inputs.Length), Position(toX, area, b, outputs.Length));
            }
        }
    }

    /// <summary>Индексы нейронов для отображения: все или равномерная выборка с первым и последним.</summary>
    private static int[] SampleNeurons(int count)
    {
        int shown = Math.Min(count, MaxNeuronsPerColumn);
        var indices = new int[shown];
        for (int k = 0; k < shown; k++)
        {
            indices[k] = shown == 1 ? 0 : (int)Math.Round(k * (count - 1) / (double)(shown - 1));
        }

        return indices;
    }

    private static PointF Position(float x, Rectangle area, int index, int count) =>
        new(x, area.Top + (count == 1 ? area.Height / 2f : index * (area.Height - 1) / (float)(count - 1)));
}
