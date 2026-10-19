using System;
using System.Drawing;
using System.Linq;
using System.Text.Json;
using System.Xml.Linq;
using IrisNeuralNet.Core;
using IrisNeuralNet.Lab.Rendering;
using IrisNeuralNet.Lab.Training;
using Xunit;

namespace IrisNeuralNet.Tests;

public sealed class RenderingTests
{
    private static readonly XNamespace Svg = "http://www.w3.org/2000/svg";

    [Fact]
    public void ChartPainter_ProducesWellFormedSvgWithLegendAndHoverData()
    {
        var svg = new SvgBuilder(700f, 300f);
        ChartPainter.Draw(svg, new RectangleF(0f, 0f, 700f, 300f), "Loss <train & val>", new[]
        {
            new ChartSeries("train", Theme.AccentBlue, new[] { 1.0f, 0.5f, 0.25f }),
            new ChartSeries("val", Theme.AccentOrange, new[] { 1.1f, 0.6f, 0.4f }),
        }, ChartStyle.Dark);

        XElement root = XElement.Parse(svg.ToString());
        string[] texts = root.Descendants(Svg + "text").Select(t => t.Value).ToArray();
        Assert.Contains("Loss <train & val>", texts);
        Assert.Contains("train", texts);
        Assert.Contains("0.4000", texts);
        Assert.Equal(2, root.Descendants(Svg + "polyline").Count());

        using JsonDocument hover = JsonDocument.Parse(root.Attribute("data-hover")!.Value);
        Assert.Equal(3, hover.RootElement.GetProperty("count").GetInt32());
        Assert.Equal(2, hover.RootElement.GetProperty("series").GetArrayLength());
    }

    [Fact]
    public void ChartPainter_SkipsNaNSeriesAndShowsWaitingWithoutData()
    {
        var svg = new SvgBuilder(600f, 300f);
        ChartPainter.Draw(svg, new RectangleF(0f, 0f, 600f, 300f), "Loss", new[]
        {
            new ChartSeries("train", Theme.AccentBlue, new[] { 1f, 0.5f }),
            new ChartSeries("val", Theme.AccentOrange, new[] { float.NaN, float.NaN }),
        }, ChartStyle.Dark);
        XElement withNaN = XElement.Parse(svg.ToString());
        Assert.Single(withNaN.Descendants(Svg + "polyline"));
        Assert.Contains("null", withNaN.Attribute("data-hover")!.Value);

        var empty = new SvgBuilder(600f, 300f);
        ChartPainter.Draw(empty, new RectangleF(0f, 0f, 600f, 300f), "Loss", Array.Empty<ChartSeries>(), ChartStyle.Dark);
        Assert.Contains(XElement.Parse(empty.ToString()).Descendants(Svg + "text"), t => t.Value == "ожидание данных...");
    }

    [Fact]
    public void ChartPainter_ThinsLongSeriesButKeepsExtremesAndLastPoint()
    {
        float[] values = Enumerable.Range(0, 5000).Select(i => MathF.Sin(i / 50f)).ToArray();
        values[1234] = 5f; // выброс должен остаться на графике

        var svg = new SvgBuilder(700f, 300f);
        ChartPainter.Draw(svg, new RectangleF(0f, 0f, 700f, 300f), "Loss", new[] { new ChartSeries("train", Theme.AccentBlue, values) }, ChartStyle.Light);

        XElement polyline = XElement.Parse(svg.ToString()).Descendants(Svg + "polyline").Single();
        string[] points = polyline.Attribute("points")!.Value.Split(' ');
        Assert.InRange(points.Length, 100, 2000);
        float[] ys = points.Select(p => float.Parse(p.Split(',')[1], System.Globalization.CultureInfo.InvariantCulture)).ToArray();
        Assert.Equal(30f, ys.Min(), precision: 0); // максимум ряда — у верхнего края графика
    }

    [Fact]
    public void ReportCharts_AreValidXmlWithTwoPanelsAndNoHoverData()
    {
        string path = System.IO.Path.GetTempFileName();
        try
        {
            var history = new IrisNeuralNet.Diagnostics.TrainingHistory();
            history.OnEpochCompleted(new IrisNeuralNet.Diagnostics.EpochMetrics(1, 1f, 0.5f, 1.2f, 0.4f));
            history.OnEpochCompleted(new IrisNeuralNet.Diagnostics.EpochMetrics(2, 0.5f, 0.8f, 0.7f, 0.7f));

            new SvgChartRenderer().SaveHistory(path, "Модель", history, "Accuracy");

            XElement root = XElement.Load(path);
            Assert.Null(root.Attribute("data-hover"));
            Assert.Equal(4, root.Descendants(Svg + "polyline").Count());
            Assert.Contains(root.Descendants(Svg + "text"), t => t.Value == "Модель — Accuracy");
        }
        finally
        {
            System.IO.File.Delete(path);
        }
    }

    [Fact]
    public void Theme_ToCss_ExposesPaletteVariables()
    {
        string css = Theme.ToCss();

        Assert.Contains("--background: #181a1f;", css);
        Assert.Contains("--accent-green: #5ec979;", css);
        Assert.Contains("--font-mono:", css);
    }

    [Fact]
    public void Topology_SamplesWideLayersEvenlyIncludingEnds()
    {
        int[] sampled = TopologyPainter.SampleNeurons(784);

        Assert.Equal(24, sampled.Length);
        Assert.Equal(0, sampled[0]);
        Assert.Equal(783, sampled[^1]);
        Assert.Equal(sampled.OrderBy(i => i), sampled);
        Assert.Equal(new[] { 0, 1, 2 }, TopologyPainter.SampleNeurons(3));
    }

    [Fact]
    public void Topology_DrawsNetworkWithoutTouchingTrainingCaches()
    {
        NeuralNetwork network = NetworkFactory.Create(784, new[] { 64 }, TrainingTask.MultiClass, classCount: 10, seed: 1);
        var svg = new SvgBuilder(500f, 400f);

        TopologyPainter.Draw(svg, new RectangleF(0f, 0f, 500f, 400f), network, new float[784]);

        XElement root = XElement.Parse(svg.ToString());
        Assert.Equal(24 + 24 + 10, root.Descendants(Svg + "circle").Count());
        Assert.Contains(root.Descendants(Svg + "text"), t => t.Value == "показано 24");
    }

    [Fact]
    public void SweepBest_DoesNotDependOnCompletionOrder()
    {
        SweepResult[] results =
        {
            new(2, 32, 1f, 1f, 0.05f, 10),
            new(0, 2, 1f, 1f, 0.05f, 10),
            new(1, 8, 1f, 0.9f, 0.01f, 10),
            new(3, 2, float.NaN, float.NaN, float.NaN, 10),
        };

        SweepResult best = SweepRunner.Best(results, Metric.Accuracy);
        SweepResult reversed = SweepRunner.Best(results.Reverse(), Metric.Accuracy);

        Assert.Equal(best, reversed);
        Assert.Equal((0, 2), (best.HiddenLayers, best.Neurons));
        SweepResult lowestError = SweepRunner.Best(results, Metric.MeanAbsoluteError);
        Assert.Equal((1, 8), (lowestError.HiddenLayers, lowestError.Neurons));
    }

    [Fact]
    public void SvgBuilder_EscapesTextAndFormatsNumbersInvariantly()
    {
        var svg = new SvgBuilder(10.26f, 20f).Text(1.06f, 2f, "a \"b\" <c>", Color.Black).Data("x", "{\"k\":1}");

        string text = svg.ToString();
        XElement root = XElement.Parse(text);
        Assert.Equal("a \"b\" <c>", root.Descendants(Svg + "text").Single().Value);
        Assert.Equal("{\"k\":1}", root.Attribute("data-x")!.Value);
        Assert.Contains("width=\"10.3\"", text);
        Assert.DoesNotContain(",5", text);
    }
}
