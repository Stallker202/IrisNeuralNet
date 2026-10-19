using System;
using System.Diagnostics;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Xml.Linq;
using IrisNeuralNet.Core;
using IrisNeuralNet.Data;
using IrisNeuralNet.Lab.Experiments;
using IrisNeuralNet.Lab.Web;
using IrisNeuralNet.Optimizers;
using Microsoft.AspNetCore.Builder;
using Xunit;

namespace IrisNeuralNet.Tests;

public sealed class ArchitectureParserTests
{
    [Theory]
    [InlineData("512, 256", new[] { 512, 256 })]
    [InlineData("60->15", new[] { 60, 15 })]
    [InlineData("8 → 4", new[] { 8, 4 })]
    [InlineData("-", new int[0])]
    [InlineData("  ", new int[0])]
    public void TryParse_AcceptsCommonNotations(string text, int[] expected)
    {
        Assert.True(ArchitectureParser.TryParse(text, out int[] hidden, out _));
        Assert.Equal(expected, hidden);
    }

    [Theory]
    [InlineData("8, x")]
    [InlineData("0")]
    [InlineData("5000")]
    [InlineData("1,1,1,1,1,1,1,1,1")]
    public void TryParse_RejectsInvalidSizes(string text)
    {
        Assert.False(ArchitectureParser.TryParse(text, out _, out string error));
        Assert.NotEqual(string.Empty, error);
    }
}

public sealed class LabStateTests
{
    private static readonly TrainRequest IrisRun = new("8", OptimizerKind.Adam, 0.02f, Epochs: 30, BatchSize: 16, EarlyStoppingPatience: 0, KeepPrevious: true);

    [Fact]
    public async Task PrepareAndTrain_ReportsEpochsAndRendersCharts()
    {
        using var lab = new LabState();
        Assert.Null(await lab.PrepareAsync(new DataRequest("iris", FeatureScaling.Standardize, 0.2f, 0.02f)));

        StateDto ready = lab.Snapshot(Idle());
        Assert.True(ready.Ready);
        Assert.Contains("жми Start", ready.Status);
        Assert.True(ready.SupportsProjection);
        Assert.False(ready.SupportsDigits);

        Assert.Null(lab.Start(IrisRun));
        StateDto done = await WaitUntilIdle(lab);
        Assert.StartsWith("epoch 30 |", done.Status);
        Assert.True(done.Revision > ready.Revision);

        XElement loss = XElement.Parse(lab.RenderHistoryChart(isLoss: true, 700f, 300f, overlay: true));
        Assert.Equal(2, loss.Descendants().Count(e => e.Name.LocalName == "polyline"));
        string boundary = lab.RenderBoundary(500f, 400f);
        Assert.Contains("Границы решений", boundary);
        Assert.Equal(150, XElement.Parse(boundary).Descendants().Count(e => e.Name.LocalName == "circle"));
        Assert.Contains("L1 Relu 8", lab.RenderTopology(500f, 400f));
    }

    [Fact]
    public async Task Start_RejectsInvalidSettingsWithReadableErrors()
    {
        using var lab = new LabState();
        Assert.Equal("данные ещё не готовы", lab.Start(IrisRun));

        await lab.PrepareAsync(new DataRequest("iris", FeatureScaling.None, 0.2f, 0.01f));
        Assert.Contains("«x»", lab.Start(IrisRun with { Hidden = "8, x" }));
        Assert.Contains("эпох", lab.Start(IrisRun with { Epochs = 0 }));
        Assert.Contains("learning rate", lab.Start(IrisRun with { LearningRate = -1f }));
        Assert.Equal("неизвестный датасет", await lab.PrepareAsync(new DataRequest("cifar", FeatureScaling.None, 0.2f, 0.01f)));
    }

    [Fact]
    public async Task SecondRun_KeepsPreviousCurvesForOverlay()
    {
        using var lab = new LabState();
        await lab.PrepareAsync(new DataRequest("iris", FeatureScaling.Standardize, 0.2f, 0.02f));
        lab.Start(IrisRun);
        await WaitUntilIdle(lab);
        lab.Start(IrisRun with { Hidden = "4" });
        await WaitUntilIdle(lab);

        string chart = lab.RenderHistoryChart(isLoss: false, 700f, 300f, overlay: true);

        Assert.Contains("train (prev)", chart);
        Assert.DoesNotContain("train (prev)", lab.RenderHistoryChart(isLoss: false, 700f, 300f, overlay: false));
    }

    [Fact]
    public async Task Sweep_RunsWholeGridAndResetsWhenDataChanges()
    {
        using var lab = new LabState();
        using var sweep = new SweepCoordinator(lab);
        await lab.PrepareAsync(new DataRequest("iris", FeatureScaling.Standardize, 0.2f, 0.02f));

        Assert.Null(sweep.Start(new SweepRequest(IrisRun, Epochs: 3)));
        SweepStateDto finished = await WaitFor(() => sweep.Snapshot(), s => !s.Running);
        Assert.Equal(SweepCoordinator.Grid.Count, finished.Done);
        Assert.StartsWith("лучшая:", finished.Status);
        Assert.Contains("слоёв: 3", sweep.Render(800f, 500f));

        await lab.PrepareAsync(new DataRequest("iris", FeatureScaling.None, 0.2f, 0.02f));
        Assert.Equal(0, sweep.Snapshot().Done);
        Assert.Contains("Запустить sweep", sweep.Render(800f, 500f));
    }

    [Fact]
    public void DigitService_PredictsMnistShapedInput()
    {
        var random = new Random(3);
        float[] Pixels(int rows) => Enumerable.Range(0, rows * 784).Select(_ => (float)random.Next(256)).ToArray();
        float[] Labels(int rows) => Enumerable.Range(0, rows).Select(i => (float)(i % 10)).ToArray();
        PreparedData prepared = DataPreparation.PrepareSplit(
            Datasets.Mnist, new Dataset(Pixels(20), Labels(20), 784, 10), new Dataset(Pixels(10), Labels(10), 784, 10), FeatureScaling.Pixels);
        var data = new LabData(1, Datasets.Mnist, prepared, Projection: null, prepared.ValidationFeatures[..784]);
        NeuralNetwork network = NetworkFactory.Create(784, new[] { 16 }, TrainingTask.MultiClass, classCount: 10, seed: 1);

        var drawing = new byte[100 * 100];
        for (int y = 20; y < 80; y++) drawing[y * 100 + 50] = 255; // вертикальная черта — «единица»
        float[] mnist = DigitService.ToMnist(drawing, 100, 100, DigitSource.Drawing);
        DigitResultDto result = DigitService.Predict(data, network, mnist, "тест", 400f, 300f);

        Assert.True(data.SupportsDigits);
        Assert.Equal(10, result.Probabilities.Length);
        Assert.Equal(1f, result.Probabilities.Sum(), precision: 4);
        Assert.Equal(784, Convert.FromBase64String(result.Pixels).Length);
        Assert.StartsWith("data:image/png;base64,", result.Preview);
        Assert.StartsWith("это ", result.Verdict);
        Assert.Equal("тест", result.Truth);
        XElement.Parse(result.Bars);

        Assert.Throws<ArgumentException>(() => DigitService.ToMnist(new byte[10], 5, 2, DigitSource.Mnist));
        Assert.Equal("нет сети — нажми Start", DigitService.Predict(data, null, mnist, null, 400f, 300f).Verdict);
    }

    internal static SweepStateDto Idle() => new(false, 0, 0, string.Empty, 0);

    private static Task<StateDto> WaitUntilIdle(LabState lab) => WaitFor(() => lab.Snapshot(Idle()), s => !s.Running);

    internal static Task<T> WaitFor<T>(Func<T> read, Func<T, bool> done) => WaitFor(() => Task.FromResult(read()), done);

    internal static async Task<T> WaitFor<T>(Func<Task<T>> read, Func<T, bool> done, int timeoutMs = 60000)
    {
        var stopwatch = Stopwatch.StartNew();
        while (true)
        {
            T value = await read();
            if (done(value)) return value;
            if (stopwatch.ElapsedMilliseconds > timeoutMs) throw new TimeoutException("Lab did not reach the expected state.");
            await Task.Delay(20);
        }
    }
}

/// <summary>Сквозной тест HTTP: настоящий Kestrel на свободном порту, как при запуске пользователем.</summary>
public sealed class LabWebHostTests : IAsyncLifetime
{
    private WebApplication _app = null!;
    private HttpClient _http = null!;

    public async Task InitializeAsync()
    {
        _app = LabWebHost.Build(new LabHostOptions(Port: 0, OpenBrowser: false));
        await _app.StartAsync();
        _http = new HttpClient { BaseAddress = new Uri(_app.Urls.First()) };
    }

    public async Task DisposeAsync()
    {
        _http.Dispose();
        await _app.DisposeAsync();
    }

    [Fact]
    public async Task ServesPageThemeAndCatalog()
    {
        string page = await _http.GetStringAsync("/");
        Assert.Contains("<title>Iris Neural Lab</title>", page);
        Assert.Contains("--accent-green", await _http.GetStringAsync("/theme.css"));
        Assert.Contains("lab.js", page);
        Assert.Equal(HttpStatusCode.OK, (await _http.GetAsync("/lab.js")).StatusCode);

        using JsonDocument catalog = JsonDocument.Parse(await _http.GetStringAsync("/api/catalog"));
        string[] keys = catalog.RootElement.GetProperty("datasets").EnumerateArray().Select(d => d.GetProperty("key").GetString()!).ToArray();
        Assert.Equal(new[] { "iris", "sonar", "boston", "mnist" }, keys);
        Assert.Equal("RmsProp", catalog.RootElement.GetProperty("datasets")[2].GetProperty("defaults").GetProperty("optimizer").GetString());
    }

    [Fact]
    public async Task TrainsOverHttpAndReturnsSvgViews()
    {
        HttpResponseMessage prepared = await _http.PostAsJsonAsync("/api/data", new { dataset = "boston", scaling = "Standardize", validationFraction = 0.2, learningRate = 0.001 });
        Assert.Equal(HttpStatusCode.NoContent, prepared.StatusCode);

        HttpResponseMessage started = await _http.PostAsJsonAsync("/api/train/start", new
        {
            hidden = "16", optimizer = "RmsProp", learningRate = 0.001, epochs = 5, batchSize = 16, earlyStoppingPatience = 0, keepPrevious = true,
        });
        Assert.Equal(HttpStatusCode.NoContent, started.StatusCode);

        JsonElement state = await LabStateTests.WaitFor(
            async () => JsonDocument.Parse(await _http.GetStringAsync("/api/state")).RootElement,
            s => !s.GetProperty("running").GetBoolean() && s.GetProperty("status").GetString()!.StartsWith("epoch 5"));
        Assert.Contains("val mae", state.GetProperty("status").GetString());

        HttpResponseMessage svg = await _http.GetAsync("/api/view/metric.svg?w=640&h=320&overlay=true");
        Assert.Equal("image/svg+xml", svg.Content.Headers.ContentType!.MediaType);
        Assert.Contains("MAE", await svg.Content.ReadAsStringAsync());
        Assert.Equal(HttpStatusCode.NotFound, (await _http.GetAsync("/api/view/nothing.svg")).StatusCode);
    }

    [Fact]
    public async Task ReturnsReadableErrors()
    {
        HttpResponseMessage response = await _http.PostAsJsonAsync("/api/data", new { dataset = "nope", scaling = "None", validationFraction = 0.2, learningRate = 0.01 });
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Contains("неизвестный датасет", await response.Content.ReadAsStringAsync());

        HttpResponseMessage digits = await _http.PostAsync("/api/digits/predict?source=Drawing&w=2&h=2", new ByteArrayContent(new byte[4]));
        Assert.Equal(HttpStatusCode.BadRequest, digits.StatusCode);
        Assert.Contains("MNIST", await digits.Content.ReadAsStringAsync());

        using JsonDocument architecture = JsonDocument.Parse(await _http.GetStringAsync("/api/architecture?text=8,%20x"));
        Assert.False(architecture.RootElement.GetProperty("valid").GetBoolean());
    }
}
