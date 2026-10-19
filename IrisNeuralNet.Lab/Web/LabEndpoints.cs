using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using IrisNeuralNet.Lab.Experiments;
using IrisNeuralNet.Lab.Rendering;
using IrisNeuralNet.Optimizers;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;

namespace IrisNeuralNet.Lab.Web;

/// <summary>
/// HTTP-API лаборатории. Все вычисления и вся отрисовка — на сервере (C#): страница присылает
/// команды, опрашивает состояние и вставляет готовые SVG. Так один и тот же код рисует
/// и страницу, и графики отчётов.
/// </summary>
internal static class LabEndpoints
{
    private const float MinViewSize = 50f;
    private const float MaxViewSize = 4000f;
    private const string SvgContentType = "image/svg+xml; charset=utf-8";

    public static void Map(IEndpointRouteBuilder app)
    {
        app.MapGet("/theme.css", () => Results.Content(Theme.ToCss(), "text/css; charset=utf-8"));

        RouteGroupBuilder api = app.MapGroup("/api");
        api.MapGet("/catalog", Catalog);
        api.MapGet("/state", (LabState lab, SweepCoordinator sweep) => lab.Snapshot(sweep.Snapshot()));
        api.MapGet("/architecture", (string? text) =>
            ArchitectureParser.TryParse(text, out int[] hidden, out string error)
                ? new ArchitectureDto(true, string.Empty, hidden)
                : new ArchitectureDto(false, error, Array.Empty<int>()));

        api.MapPost("/data", async (DataRequest request, LabState lab, CancellationToken token) =>
            ToResult(await lab.PrepareAsync(request, token)));

        api.MapPost("/train/start", (TrainRequest request, LabState lab) => ToResult(lab.Start(request)));
        api.MapPost("/train/stop", (LabState lab) =>
        {
            lab.Stop();
            return Results.NoContent();
        });
        api.MapPut("/train/learning-rate", (LearningRateRequest request, LabState lab) => ToResult(lab.SetLearningRate(request.Value)));

        api.MapPost("/sweep/start", (SweepRequest request, SweepCoordinator sweep) => ToResult(sweep.Start(request)));
        api.MapPost("/sweep/cancel", (SweepCoordinator sweep) =>
        {
            sweep.Cancel();
            return Results.NoContent();
        });

        api.MapGet("/view/{name}.svg", View);
        api.MapPost("/digits/predict", PredictDigit);
        api.MapGet("/digits/sample", (LabState lab, float? bw, float? bh) =>
        {
            LabData? data = lab.Data;
            if (data is null || !data.SupportsDigits) return NotDigits();

            (float[] pixels, string truth) = DigitService.RandomTestSample(data);
            return Results.Ok(DigitService.Predict(data, lab.Network, pixels, truth, Size(bw), Size(bh)));
        });
    }

    private static CatalogDto Catalog() =>
        new(
            Datasets.All.Select(spec => new DatasetDto(
                spec.Key,
                spec.Name,
                spec.Task.Description,
                spec.HasPredefinedTestSplit,
                new DefaultsDto(
                    ArchitectureParser.Format(spec.Defaults.Hidden),
                    spec.Defaults.Scaling,
                    spec.Defaults.ValidationFraction,
                    spec.Defaults.Optimizer,
                    spec.Defaults.LearningRate,
                    spec.Defaults.Epochs,
                    spec.Defaults.BatchSize))).ToArray(),
            Enum.GetValues<FeatureScaling>().Select(s => new OptionDto(s.ToString(), s.DisplayName())).ToArray(),
            Enum.GetValues<OptimizerKind>()
                .Select(k => new OptimizerDto(k, OptimizerFactory.KerasName(k), OptimizerFactory.DefaultLearningRate(k)))
                .ToArray());

    private static IResult View(string name, float? w, float? h, bool? overlay, LabState lab, SweepCoordinator sweep)
    {
        float width = Size(w), height = Size(h);
        string? svg = name switch
        {
            "loss" => lab.RenderHistoryChart(isLoss: true, width, height, overlay ?? true),
            "metric" => lab.RenderHistoryChart(isLoss: false, width, height, overlay ?? true),
            "topology" => lab.RenderTopology(width, height),
            "boundary" => lab.RenderBoundary(width, height),
            "sweep" => sweep.Render(width, height),
            _ => null,
        };
        return svg is null ? Results.NotFound() : Results.Content(svg, SvgContentType);
    }

    /// <summary>Тело — байты оттенков серого (w×h, построчно): холст или файл, уже декодированный браузером.</summary>
    private static async Task<IResult> PredictDigit(
        HttpRequest http, LabState lab, DigitSource source, int w, int h, float? bw, float? bh, string? truth)
    {
        LabData? data = lab.Data;
        if (data is null || !data.SupportsDigits) return NotDigits();
        if (w is < 1 or > DigitService.MaxImageSide || h is < 1 or > DigitService.MaxImageSide)
        {
            return Error($"изображение до {DigitService.MaxImageSide}×{DigitService.MaxImageSide}");
        }

        byte[] gray = new byte[w * h];
        try
        {
            await http.Body.ReadExactlyAsync(gray, http.HttpContext.RequestAborted);
        }
        catch (EndOfStreamException)
        {
            return Error($"ожидалось {w * h} байт изображения {w}×{h}");
        }

        float[] mnist = DigitService.ToMnist(gray, w, h, source);
        return Results.Ok(DigitService.Predict(data, lab.Network, mnist, truth, Size(bw), Size(bh)));
    }

    private static float Size(float? value) => Math.Clamp(value ?? 600f, MinViewSize, MaxViewSize);

    private static IResult ToResult(string? error) => error is null ? Results.NoContent() : Error(error);

    private static IResult Error(string message) => Results.BadRequest(new ErrorResponse(message));

    private static IResult NotDigits() => Error("вкладка «Цифры» работает с MNIST: выбери его в левой панели");
}
