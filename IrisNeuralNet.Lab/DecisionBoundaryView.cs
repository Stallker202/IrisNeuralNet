using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;
using IrisNeuralNet.Core;
using IrisNeuralNet.Lab.Experiments;
using IrisNeuralNet.MathCore;

namespace IrisNeuralNet.Lab;

/// <summary>
/// Проекция данных на две главные компоненты (PCA) и ответ сети в каждой точке плоскости:
/// для классификации — класс и уверенность, для регрессии — предсказанное значение (синий → красный).
/// Для очень широких входов (MNIST, 784 признака) PCA и тепловая карта не строятся.
/// </summary>
internal sealed class DecisionBoundaryView : PictureBox
{
    private const int GridWidth = 56;
    private const int GridHeight = 40;
    private const int MaxPcaDimensions = 64;
    private const int ConfidenceLevels = 17;

    private readonly NetworkProbe _probe = new();
    private Pca2D? _pca;
    private NeuralNetwork? _network;
    private TrainingTask? _task;
    private float[] _probeBuffer = Array.Empty<float>();
    private float[] _projected = Array.Empty<float>();
    private float[] _labels = Array.Empty<float>();
    private float _labelMin;
    private float _labelMax;
    private int _dimensions;
    private string _message = "ожидание данных...";
    private SolidBrush?[,]? _cellBrushes;

    public DecisionBoundaryView()
    {
        SetStyle(ControlStyles.OptimizedDoubleBuffer | ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint, true);
        UpdateStyles();
    }

    public void SetData(PreparedData data)
    {
        _task = data.Task;
        _dimensions = data.FeatureCount;
        _pca = null;
        if (data.FeatureCount > MaxPcaDimensions)
        {
            _message = $"PCA-вид отключён: {data.FeatureCount} признаков. Для MNIST — вкладка «Цифры».";
            return;
        }

        float[] features = Concat(data.TrainFeatures, data.ValidationFeatures);
        _labels = Concat(data.RawTrain.Labels.ToArray(), data.RawValidation.Labels.ToArray());
        _pca = Pca2D.Fit(features, data.FeatureCount);
        _probeBuffer = new float[data.FeatureCount];
        _projected = new float[_labels.Length * 2];
        for (int i = 0; i < _labels.Length; i++)
        {
            _pca.Project(features.AsSpan(i * data.FeatureCount, data.FeatureCount), out _projected[i * 2], out _projected[i * 2 + 1]);
        }

        _labelMin = float.MaxValue;
        _labelMax = float.MinValue;
        foreach (float label in _labels)
        {
            _labelMin = MathF.Min(_labelMin, label);
            _labelMax = MathF.Max(_labelMax, label);
        }
    }

    public void UpdateNetwork(NeuralNetwork? network) => _network = network;

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        Graphics g = e.Graphics;
        g.SmoothingMode = SmoothingMode.AntiAlias;
        using (var background = new SolidBrush(UiTheme.ChartBackground))
        {
            g.FillRectangle(background, ClientRectangle);
        }

        Pca2D? pca = _pca;
        if (pca is null || _task is null)
        {
            using var waiting = new SolidBrush(UiTheme.TextMuted);
            g.DrawString(_message, UiTheme.UiFont, waiting, 10, 10);
            return;
        }

        using var titleBrush = new SolidBrush(UiTheme.TextPrimary);
        string title = _task.IsClassification ? "Границы решений: PCA 2D" : "Поверхность предсказаний: PCA 2D (синий — меньше, красный — больше)";
        g.DrawString(title, UiTheme.TitleFont, titleBrush, 8, 6);

        Rectangle plot = Rectangle.FromLTRB(40, 30, ClientRectangle.Right - 12, ClientRectangle.Bottom - 30);
        if (plot.Width < 20 || plot.Height < 20) return;

        NeuralNetwork? network = _network;
        if (network is not null && network.InputDim == _dimensions)
        {
            DrawHeatmap(g, plot, pca, network);
        }

        DrawSamples(g, plot, pca);

        using var frame = new Pen(UiTheme.Border);
        g.DrawRectangle(frame, plot);
        using var muted = new SolidBrush(UiTheme.TextMuted);
        g.DrawString("PC1", UiTheme.UiFont, muted, plot.Right - 24, plot.Bottom + 8);
        g.DrawString("PC2", UiTheme.UiFont, muted, 8, plot.Top - 2);
    }

    private void DrawHeatmap(Graphics g, Rectangle plot, Pca2D pca, NeuralNetwork network)
    {
        float cellWidth = plot.Width / (float)GridWidth;
        float cellHeight = plot.Height / (float)GridHeight;
        for (int gy = 0; gy < GridHeight; gy++)
        {
            for (int gx = 0; gx < GridWidth; gx++)
            {
                float p1 = pca.P1Min + (gx + 0.5f) / GridWidth * (pca.P1Max - pca.P1Min);
                float p2 = pca.P2Max - (gy + 0.5f) / GridHeight * (pca.P2Max - pca.P2Min);
                pca.Reconstruct(p1, p2, _probeBuffer);
                ReadOnlySpan<float> output = _probe.Run(network, _probeBuffer);

                SolidBrush brush = _task is ClassificationTask classification
                    ? CellBrush(classification.PredictClass(output, out float confidence), confidence)
                    : CellBrush(ValueColorIndex(output[0]), 0.55f);
                g.FillRectangle(brush, plot.Left + gx * cellWidth, plot.Top + gy * cellHeight, cellWidth + 1, cellHeight + 1);
            }
        }
    }

    private void DrawSamples(Graphics g, Rectangle plot, Pca2D pca)
    {
        using var stroke = new Pen(UiTheme.TextPrimary, 1f);
        for (int i = 0; i < _labels.Length; i++)
        {
            float x = plot.Left + (_projected[i * 2] - pca.P1Min) / (pca.P1Max - pca.P1Min) * (plot.Width - 1);
            float y = plot.Top + (pca.P2Max - _projected[i * 2 + 1]) / (pca.P2Max - pca.P2Min) * (plot.Height - 1);
            Color color = _task!.IsClassification
                ? UiTheme.ClassColors[(int)_labels[i] % UiTheme.ClassColors.Length]
                : ValueColor(_labels[i]);
            using var fill = new SolidBrush(color);
            g.FillEllipse(fill, x - 3, y - 3, 6, 6);
            g.DrawEllipse(stroke, x - 3, y - 3, 6, 6);
        }
    }

    /// <summary>Значение регрессии → цвет между синим (минимум меток) и красным (максимум).</summary>
    private Color ValueColor(float value) =>
        UiTheme.Mix(UiTheme.AccentBlue, UiTheme.AccentRed, (value - _labelMin) / MathF.Max(_labelMax - _labelMin, 1e-6f));

    /// <summary>Индекс «класса» для кэша кистей регрессии: 0..ConfidenceLevels−1 по значению.</summary>
    private int ValueColorIndex(float value)
    {
        float t = Math.Clamp((value - _labelMin) / MathF.Max(_labelMax - _labelMin, 1e-6f), 0f, 1f);
        return UiTheme.ClassColors.Length + (int)(t * (ConfidenceLevels - 1));
    }

    private SolidBrush CellBrush(int colorIndex, float confidence)
    {
        int palette = UiTheme.ClassColors.Length;
        _cellBrushes ??= new SolidBrush?[palette + ConfidenceLevels, ConfidenceLevels];
        int level = Math.Clamp((int)(confidence * (ConfidenceLevels - 1)), 0, ConfidenceLevels - 1);
        SolidBrush? brush = _cellBrushes[colorIndex, level];
        if (brush is null)
        {
            Color color = colorIndex < palette
                ? UiTheme.ClassColors[colorIndex]
                : UiTheme.Mix(UiTheme.AccentBlue, UiTheme.AccentRed, (colorIndex - palette) / (float)(ConfidenceLevels - 1));
            brush = new SolidBrush(Color.FromArgb(26 + level * 6, color));
            _cellBrushes[colorIndex, level] = brush;
        }

        return brush;
    }

    private static float[] Concat(float[] first, float[] second)
    {
        var result = new float[first.Length + second.Length];
        first.CopyTo(result, 0);
        second.CopyTo(result, first.Length);
        return result;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing && _cellBrushes is not null)
        {
            foreach (SolidBrush? brush in _cellBrushes) brush?.Dispose();
            _cellBrushes = null;
        }

        base.Dispose(disposing);
    }
}
