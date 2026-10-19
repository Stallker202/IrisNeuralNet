using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using IrisNeuralNet.Core;
using IrisNeuralNet.Lab.Experiments;
using IrisNeuralNet.Lab.Rendering;
using IrisNeuralNet.Lab.Reporting;
using IrisNeuralNet.Lab.Training;

namespace IrisNeuralNet.Lab.Web;

/// <summary>
/// Вкладка Sweep на сервере: фоновый перебор «скрытые слои × нейроны» по текущим данным.
/// При смене данных идущий sweep отменяется, а результаты сбрасываются.
/// </summary>
internal sealed class SweepCoordinator : IDisposable
{
    public static readonly SweepGrid Grid = new(LayerOptions: new[] { 0, 1, 2, 3 }, NeuronOptions: new[] { 2, 4, 8, 16, 32, 64 });

    private const string IdleStatus = "ещё не запущено (оптимизатор, lr и batch — из левой панели)";

    private readonly LabState _lab;
    private readonly object _gate = new();
    private readonly List<SweepResult> _results = new();
    private LabData? _data;
    private CancellationTokenSource? _cancellation;
    private Task? _run;
    private bool _started;
    private long _revision;
    private string _status = IdleStatus;

    public SweepCoordinator(LabState lab)
    {
        _lab = lab ?? throw new ArgumentNullException(nameof(lab));
        _lab.DataChanged += Reset;
        _data = lab.Data;
    }

    private bool RunningLocked => _run is { IsCompleted: false };

    /// <returns>null — sweep запущен; иначе причина отказа.</returns>
    public string? Start(SweepRequest request)
    {
        if (request is null) return "пустой запрос";
        if (request.Epochs is < 1 or > RequestValidation.MaxEpochs) return $"эпох — от 1 до {RequestValidation.MaxEpochs}";

        lock (_gate)
        {
            if (_data is null) return "данные ещё не готовы";
            if (RunningLocked) return "sweep уже идёт";
            if (!RequestValidation.TryCreateConfig(request.Template, _data.Spec, out ExperimentConfig config, out string error)) return error;

            LabData data = _data;
            var cancellation = new CancellationTokenSource();
            _cancellation?.Dispose();
            _cancellation = cancellation;
            _results.Clear();
            _started = true;
            _status = $"sweep: 0/{Grid.Count}";
            _revision++;
            _run = Task.Run(() => Execute(data, config with { Epochs = request.Epochs }, cancellation.Token));
            return null;
        }
    }

    public void Cancel()
    {
        lock (_gate) _cancellation?.Cancel();
    }

    public SweepStateDto Snapshot()
    {
        lock (_gate) return new SweepStateDto(RunningLocked, _results.Count, Grid.Count, _status, _revision);
    }

    public string Render(float width, float height)
    {
        SweepResult[] results;
        bool started;
        Metric metric;
        lock (_gate)
        {
            results = _results.ToArray();
            started = _started;
            metric = _data?.Prepared.Task.Metric ?? Metric.Accuracy;
        }

        var svg = new SvgBuilder(width, height);
        SweepHeatmapPainter.Draw(svg, new RectangleF(0f, 0f, width, height), started ? Grid : null, results, metric);
        return svg.ToString();
    }

    public void Dispose()
    {
        _lab.DataChanged -= Reset;
        Task? run;
        lock (_gate)
        {
            _cancellation?.Cancel();
            run = _run;
        }

        try
        {
            run?.Wait(TimeSpan.FromSeconds(5));
        }
        catch (AggregateException)
        {
            // Отмена при выключении — штатный путь.
        }
    }

    private void Execute(LabData data, ExperimentConfig template, CancellationToken token)
    {
        var progress = new InlineProgress<SweepResult>(result =>
        {
            lock (_gate)
            {
                if (!ReferenceEquals(data, _data)) return;
                _results.Add(result);
                _status = $"sweep: {_results.Count}/{Grid.Count}";
                _revision++;
            }
        });

        string status;
        try
        {
            List<SweepResult> results = new SweepRunner(data.Prepared).Run(Grid, template, progress, token);
            Metric metric = data.Prepared.Task.Metric;
            SweepResult best = SweepRunner.Best(results, metric);
            status = $"лучшая: {best.HiddenLayers} скрытых × {best.Neurons} нейронов → val {metric.Name} " +
                     $"{MetricFormat.Value(metric, best.ValidationMetric)}, val_loss {MetricFormat.Loss(best.ValidationLoss)}; прогонов: {results.Count}";
        }
        catch (Exception e) when (IsCancellation(e))
        {
            status = $"sweep отменён: готово {CountFor(data)}/{Grid.Count}";
        }
        catch (Exception e)
        {
            status = $"ошибка sweep: {e.Message}";
        }

        lock (_gate)
        {
            if (!ReferenceEquals(data, _data)) return;
            _status = status;
            _revision++;
        }
    }

    private int CountFor(LabData data)
    {
        lock (_gate) return ReferenceEquals(data, _data) ? _results.Count : 0;
    }

    private void Reset(LabData? data)
    {
        lock (_gate)
        {
            _cancellation?.Cancel();
            _data = data;
            _results.Clear();
            _started = false;
            _status = IdleStatus;
            _revision++;
        }
    }

    /// <summary>Parallel.ForEach заворачивает отмену из тела цикла в AggregateException.</summary>
    private static bool IsCancellation(Exception e) =>
        e is OperationCanceledException
        || e is AggregateException aggregate && aggregate.Flatten().InnerExceptions.All(inner => inner is OperationCanceledException);
}
