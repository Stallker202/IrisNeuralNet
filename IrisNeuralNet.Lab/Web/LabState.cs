using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.Threading;
using System.Threading.Tasks;
using IrisNeuralNet.Core;
using IrisNeuralNet.Diagnostics;
using IrisNeuralNet.Lab.Experiments;
using IrisNeuralNet.Lab.Rendering;
using IrisNeuralNet.Lab.Reporting;
using IrisNeuralNet.Lab.Training;

namespace IrisNeuralNet.Lab.Web;

/// <summary>Подготовленные данные текущего датасета и всё, что из них вычисляется один раз.</summary>
internal sealed record LabData(int Version, DatasetSpec Spec, PreparedData Prepared, ProjectionModel? Projection, float[] ProbeSample)
{
    public bool SupportsDigits =>
        Prepared.FeatureCount == DigitService.Side * DigitService.Side && Prepared.ClassCount == DigitService.ClassCount;

    public string Description =>
        $"{Spec.Name}: {Prepared.TrainRowCount} train / {Prepared.ValidationRowCount} val, {Prepared.FeatureCount} признаков | " +
        $"задача: {Prepared.Task.Description} | признаки: {Prepared.Scaling.DisplayName()} — жми Start";
}

/// <summary>
/// Состояние лаборатории на сервере (то, что раньше жило в окне WinForms): текущие данные,
/// фоновый прогон обучения, история кривых и строка статуса. Приложение локальное, поэтому
/// состояние одно на процесс: несколько вкладок браузера видят одну и ту же лабораторию.
/// <para>
/// Синхронизация: <c>_gate</c> — короткие чтения и записи полей; <c>_control</c> — команды,
/// которые ждут остановки прогона (старт, стоп, смена данных), чтобы опросы страницы не ждали вместе с ними.
/// </para>
/// </summary>
internal sealed class LabState : IDisposable
{
    private readonly object _gate = new();
    private readonly SemaphoreSlim _control = new(1, 1);
    private readonly RunHistory _history = new();
    private LabData? _data;
    private TrainingSession? _session;
    private int _version;
    private long _revision;
    private bool _hasRun;
    private string _status = "выбери датасет";

    /// <summary>Данные сменились (null — идёт подготовка новых): sweep по старым данным теряет смысл.</summary>
    public event Action<LabData?>? DataChanged;

    public LabData? Data
    {
        get { lock (_gate) return _data; }
    }

    public NeuralNetwork? Network
    {
        get { lock (_gate) return _session?.CurrentNetwork; }
    }

    /// <returns>null — успех или подготовку перебил более новый запрос; иначе текст ошибки.</returns>
    public async Task<string?> PrepareAsync(DataRequest request, CancellationToken token = default)
    {
        DatasetSpec? spec = request is null ? null : Datasets.Find(request.Dataset);
        if (spec is null) return "неизвестный датасет";
        if (!Enum.IsDefined(request!.Scaling)) return "неизвестная предобработка";
        if (!spec.HasPredefinedTestSplit && request.ValidationFraction is < 0.05f or > 0.5f) return "доля validation — от 0.05 до 0.5";
        if (!RequestValidation.IsLearningRate(request.LearningRate)) return "learning rate должен быть больше 0 и не больше 10";

        int version = await DetachSessionAsync($"{spec.Name}: загрузка и подготовка данных…").ConfigureAwait(false);
        DataChanged?.Invoke(null);
        try
        {
            if (!spec.IsAvailable())
            {
                SetStatus(version, spec.MissingHint);
                if (spec.DownloadAsync is null) return spec.MissingHint;

                await spec.DownloadAsync(new InlineProgress<string>(message => SetStatus(version, message)), token).ConfigureAwait(false);
                SetStatus(version, $"{spec.Name}: загрузка и подготовка данных…");
            }

            LabData data = await Task.Run(() => Load(version, spec, request), token).ConfigureAwait(false);
            lock (_gate)
            {
                if (version != _version) return null;

                _data = data;
                _session = new TrainingSession(data.Prepared, request.LearningRate);
                _status = data.Description;
                _revision++;
            }

            DataChanged?.Invoke(data);
            return null;
        }
        catch (Exception e) when (e is not OperationCanceledException)
        {
            string error = $"{spec.Name}: ошибка подготовки данных — {e.Message}";
            SetStatus(version, error);
            return error;
        }
    }

    /// <returns>null — прогон запущен; иначе причина отказа.</returns>
    public string? Start(TrainRequest request)
    {
        LabData? data;
        TrainingSession? session;
        lock (_gate)
        {
            data = _data;
            session = _session;
        }

        if (data is null || session is null) return "данные ещё не готовы";
        if (!RequestValidation.TryCreateConfig(request, data.Spec, out ExperimentConfig config, out string error)) return error;

        _control.Wait();
        try
        {
            session.Stop();
            lock (_gate)
            {
                if (!ReferenceEquals(_session, session)) return "данные сменились — нажми Start ещё раз";

                // Под одной блокировкой с опросами: эпохи старого прогона не попадут в историю нового.
                _history.StartNewRun(keepPrevious: request.KeepPrevious);
                session.LearningRate = config.LearningRate;
                session.Start(config);
                _hasRun = true;
                _revision++;
            }

            return null;
        }
        finally
        {
            _control.Release();
        }
    }

    public void Stop()
    {
        _control.Wait();
        try
        {
            TrainingSession? session;
            lock (_gate) session = _session;
            session?.Stop();
        }
        finally
        {
            _control.Release();
        }
    }

    /// <summary>Live-изменение learning rate: работающий оптимизатор подхватит его на следующем шаге.</summary>
    public string? SetLearningRate(float value)
    {
        if (!RequestValidation.IsLearningRate(value)) return "learning rate должен быть больше 0 и не больше 10";

        lock (_gate)
        {
            if (_session is not null) _session.LearningRate = value;
        }

        return null;
    }

    public StateDto Snapshot(SweepStateDto sweep)
    {
        lock (_gate)
        {
            ConsumeLocked();
            bool running = _session?.IsRunning ?? false;
            return new StateDto(
                DataVersion: _data?.Version ?? 0,
                Dataset: _data?.Spec.Key,
                Ready: _session is not null,
                Running: running,
                Revision: _revision,
                Status: StatusLocked(running),
                LearningRate: _session?.LearningRate ?? 0f,
                SupportsProjection: _data?.Projection is not null,
                SupportsDigits: _data?.SupportsDigits ?? false,
                Sweep: sweep);
        }
    }

    /// <summary>График потерь или метрики с оверлеем прошлого прогона (как в окне WinForms).</summary>
    public string RenderHistoryChart(bool isLoss, float width, float height, bool overlay)
    {
        lock (_gate)
        {
            ConsumeLocked();
            var series = new List<ChartSeries>(4);
            if (overlay && _history.HasPrevious)
            {
                RunSeries previous = _history.Previous;
                series.Add(new ChartSeries("train (prev)", Theme.PreviousRun, isLoss ? previous.TrainLoss : previous.TrainMetric));
                series.Add(new ChartSeries("val (prev)", Theme.PreviousRunDim, isLoss ? previous.ValidationLoss : previous.ValidationMetric));
            }

            RunSeries current = _history.Current;
            series.Add(new ChartSeries("train", isLoss ? Theme.AccentBlue : Theme.AccentGreen, isLoss ? current.TrainLoss : current.TrainMetric));
            series.Add(new ChartSeries("val", isLoss ? Theme.AccentOrange : Theme.AccentGold, isLoss ? current.ValidationLoss : current.ValidationMetric));

            string title = isLoss ? "Loss" : _data is null ? "Metric" : MetricFormat.Label(_data.Prepared.Task.Metric);
            var svg = new SvgBuilder(width, height);
            ChartPainter.Draw(svg, new RectangleF(0f, 0f, width, height), title, series, ChartStyle.Dark);
            return svg.ToString();
        }
    }

    public string RenderTopology(float width, float height)
    {
        LabData? data = Data;
        var svg = new SvgBuilder(width, height);
        TopologyPainter.Draw(svg, new RectangleF(0f, 0f, width, height), Network, data?.ProbeSample ?? Array.Empty<float>());
        return svg.ToString();
    }

    public string RenderBoundary(float width, float height)
    {
        LabData? data = Data;
        string message = data is null
            ? "ожидание данных..."
            : $"PCA-вид отключён: {data.Prepared.FeatureCount} признаков. Для MNIST — вкладка «Цифры».";
        var svg = new SvgBuilder(width, height);
        BoundaryPainter.Draw(svg, new RectangleF(0f, 0f, width, height), data?.Projection, Network, message);
        return svg.ToString();
    }

    public void Dispose()
    {
        TrainingSession? session;
        lock (_gate)
        {
            session = _session;
            _session = null;
        }

        session?.Dispose();
        _control.Dispose();
    }

    private static LabData Load(int version, DatasetSpec spec, DataRequest request)
    {
        PreparedData prepared = DataPreparation.Prepare(spec, request.Scaling, request.ValidationFraction);
        float[] probe = prepared.ValidationFeatures.AsSpan(0, prepared.FeatureCount).ToArray();
        return new LabData(version, spec, prepared, ProjectionModel.TryCreate(prepared), probe);
    }

    /// <summary>Отцепляет текущую сессию (дождавшись остановки прогона) и открывает новую версию данных.</summary>
    private async Task<int> DetachSessionAsync(string status)
    {
        await _control.WaitAsync().ConfigureAwait(false);
        try
        {
            TrainingSession? old;
            int version;
            lock (_gate)
            {
                version = ++_version;
                old = _session;
                _session = null;
                _data = null;
                _history.Reset();
                _hasRun = false;
                _status = status;
                _revision++;
            }

            old?.Dispose();
            return version;
        }
        finally
        {
            _control.Release();
        }
    }

    private void SetStatus(int version, string status)
    {
        lock (_gate)
        {
            if (version == _version) _status = status;
        }
    }

    /// <summary>Забирает новые эпохи из кольца метрик; графикам есть что перерисовать — растёт ревизия.</summary>
    private void ConsumeLocked()
    {
        if (_session is null) return;

        int before = _history.Current.Count;
        _history.Consume(_session.Ring);
        if (_history.Current.Count != before) _revision++;
    }

    private string StatusLocked(bool running)
    {
        if (_session is null || _data is null || !_hasRun) return _status;

        EpochMetrics latest = _history.Latest;
        Metric metric = _data.Prepared.Task.Metric;
        bool hasEpoch = latest.Epoch > 0;
        string state = _session.LastError is { } error ? $"ошибка: {error.Message}" : running ? "обучение" : "простой";
        return $"epoch {latest.Epoch} | lr {_session.LearningRate.ToString("0.0000", CultureInfo.InvariantCulture)} | " +
               $"train {metric.Name} {MetricFormat.Value(metric, hasEpoch ? latest.TrainMetric : float.NaN)} | " +
               $"val {metric.Name} {MetricFormat.Value(metric, hasEpoch ? latest.ValidationMetric : float.NaN)} | " +
               $"val_loss {MetricFormat.Loss(hasEpoch ? latest.ValidationLoss : float.NaN)} | {state}";
    }
}

/// <summary>IProgress без контекста синхронизации: вызывает обработчик сразу, в потоке отправителя.</summary>
internal sealed class InlineProgress<T> : IProgress<T>
{
    private readonly Action<T> _handler;

    public InlineProgress(Action<T> handler) => _handler = handler ?? throw new ArgumentNullException(nameof(handler));

    public void Report(T value) => _handler(value);
}
