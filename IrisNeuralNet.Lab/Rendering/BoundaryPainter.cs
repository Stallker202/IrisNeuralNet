using System;
using System.Drawing;
using IrisNeuralNet.Core;
using IrisNeuralNet.Lab.Experiments;
using IrisNeuralNet.Lab.Training;
using IrisNeuralNet.MathCore;

namespace IrisNeuralNet.Lab.Rendering;

/// <summary>
/// Проекция данных на две главные компоненты: считается один раз при подготовке данных,
/// затем на каждом кадре по ней рисуется ответ текущей сети. Для очень широких входов
/// (MNIST, 784 признака) не строится.
/// </summary>
internal sealed class ProjectionModel
{
    public const int MaxDimensions = 64;

    private ProjectionModel(Pca2D pca, TrainingTask task, int dimensions, float[] projected, float[] labels)
    {
        Pca = pca;
        Task = task;
        Dimensions = dimensions;
        Projected = projected;
        Labels = labels;
        LabelMin = float.MaxValue;
        LabelMax = float.MinValue;
        foreach (float label in labels)
        {
            LabelMin = MathF.Min(LabelMin, label);
            LabelMax = MathF.Max(LabelMax, label);
        }
    }

    public Pca2D Pca { get; }
    public TrainingTask Task { get; }
    public int Dimensions { get; }

    /// <summary>Координаты образцов в плоскости PC1×PC2: [p1, p2, p1, p2, …].</summary>
    public float[] Projected { get; }

    public float[] Labels { get; }
    public float LabelMin { get; }
    public float LabelMax { get; }

    public static ProjectionModel? TryCreate(PreparedData data)
    {
        if (data.FeatureCount > MaxDimensions || data.FeatureCount < 2) return null;

        float[] features = Concat(data.TrainFeatures, data.ValidationFeatures);
        float[] labels = Concat(data.RawTrain.Labels.ToArray(), data.RawValidation.Labels.ToArray());
        Pca2D pca = Pca2D.Fit(features, data.FeatureCount);
        var projected = new float[labels.Length * 2];
        for (int i = 0; i < labels.Length; i++)
        {
            pca.Project(features.AsSpan(i * data.FeatureCount, data.FeatureCount), out projected[i * 2], out projected[i * 2 + 1]);
        }

        return new ProjectionModel(pca, data.Task, data.FeatureCount, projected, labels);
    }

    /// <summary>Значение регрессии → 0..1 относительно диапазона меток.</summary>
    public float Normalize(float value) => Math.Clamp((value - LabelMin) / MathF.Max(LabelMax - LabelMin, 1e-6f), 0f, 1f);

    private static float[] Concat(float[] first, float[] second)
    {
        var result = new float[first.Length + second.Length];
        first.CopyTo(result, 0);
        second.CopyTo(result, first.Length);
        return result;
    }
}

/// <summary>
/// Границы решений в плоскости PCA: для классификации — класс и уверенность в каждой клетке,
/// для регрессии — предсказанное значение (синий — меньше, красный — больше); поверх — образцы.
/// </summary>
internal static class BoundaryPainter
{
    private const int GridWidth = 56;
    private const int GridHeight = 40;
    private const int ConfidenceLevels = 17;

    public static void Draw(SvgBuilder svg, RectangleF bounds, ProjectionModel? model, NeuralNetwork? network, string emptyMessage)
    {
        svg.Rect(bounds, Theme.ChartBackground);
        if (model is null)
        {
            svg.Text(10f, 10f, emptyMessage, Theme.TextMuted);
            return;
        }

        svg.Text(8f, 6f, model.Task.IsClassification ? "Границы решений: PCA 2D" : "Поверхность предсказаний: PCA 2D", Theme.TextPrimary, TextRole.Title);

        RectangleF plot = RectangleF.FromLTRB(40f, 30f, bounds.Right - 12f, bounds.Bottom - 30f);
        if (plot.Width < 20f || plot.Height < 20f) return;

        if (network is not null && network.InputDim == model.Dimensions)
        {
            DrawHeatmap(svg, plot, model, network);
        }

        DrawSamples(svg, plot, model);
        svg.Rect(plot, fill: null, stroke: Theme.Border);
        svg.Text(plot.Right - 24f, plot.Bottom + 8f, "PC1", Theme.TextMuted);
        if (!model.Task.IsClassification)
        {
            svg.Text(plot.Left, plot.Bottom + 8f, "синий — меньше, красный — больше", Theme.TextMuted);
        }
        svg.Text(8f, plot.Top - 2f, "PC2", Theme.TextMuted);
    }

    private static void DrawHeatmap(SvgBuilder svg, RectangleF plot, ProjectionModel model, NeuralNetwork network)
    {
        Pca2D pca = model.Pca;
        var probe = new NetworkProbe();
        var sample = new float[model.Dimensions];
        float cellWidth = plot.Width / GridWidth;
        float cellHeight = plot.Height / GridHeight;

        for (int gy = 0; gy < GridHeight; gy++)
        {
            // Соседние клетки одного цвета сливаются в один прямоугольник — SVG в разы легче.
            Color? runColor = null;
            int runStart = 0;
            for (int gx = 0; gx <= GridWidth; gx++)
            {
                Color? color = null;
                if (gx < GridWidth)
                {
                    float p1 = pca.P1Min + (gx + 0.5f) / GridWidth * (pca.P1Max - pca.P1Min);
                    float p2 = pca.P2Max - (gy + 0.5f) / GridHeight * (pca.P2Max - pca.P2Min);
                    pca.Reconstruct(p1, p2, sample);
                    color = CellColor(model, probe.Run(network, sample));
                }

                if (color == runColor) continue;
                if (runColor is { } fill)
                {
                    svg.Rect(plot.Left + runStart * cellWidth, plot.Top + gy * cellHeight, (gx - runStart) * cellWidth + 0.6f, cellHeight + 0.6f, fill);
                }

                runColor = color;
                runStart = gx;
            }
        }
    }

    /// <summary>Цвет клетки квантуется по уверенности (17 уровней): соседние клетки сливаются в полосы одного цвета.</summary>
    private static Color CellColor(ProjectionModel model, ReadOnlySpan<float> output)
    {
        Color baseColor;
        float strength;
        if (model.Task is ClassificationTask classification)
        {
            int cls = classification.PredictClass(output, out float confidence);
            baseColor = Theme.ClassColors[cls % Theme.ClassColors.Length];
            strength = float.IsNaN(confidence) ? 0f : confidence;
        }
        else
        {
            float t = MathF.Round(model.Normalize(output[0]) * (ConfidenceLevels - 1)) / (ConfidenceLevels - 1);
            baseColor = Theme.Mix(Theme.AccentBlue, Theme.AccentRed, t);
            strength = 0.55f;
        }

        // Прозрачность смешивается с фоном заранее: непрозрачные соседние клетки не дают полос на стыках.
        int level = Math.Clamp((int)(strength * (ConfidenceLevels - 1)), 0, ConfidenceLevels - 1);
        return Theme.Mix(Theme.ChartBackground, baseColor, (26 + level * 6) / 255f);
    }

    private static void DrawSamples(SvgBuilder svg, RectangleF plot, ProjectionModel model)
    {
        Pca2D pca = model.Pca;
        for (int i = 0; i < model.Labels.Length; i++)
        {
            float x = plot.Left + (model.Projected[i * 2] - pca.P1Min) / (pca.P1Max - pca.P1Min) * (plot.Width - 1f);
            float y = plot.Top + (pca.P2Max - model.Projected[i * 2 + 1]) / (pca.P2Max - pca.P2Min) * (plot.Height - 1f);
            Color color = model.Task.IsClassification
                ? Theme.ClassColors[(int)model.Labels[i] % Theme.ClassColors.Length]
                : Theme.Mix(Theme.AccentBlue, Theme.AccentRed, model.Normalize(model.Labels[i]));
            svg.Circle(x, y, 3f, color, Theme.TextPrimary);
        }
    }
}
