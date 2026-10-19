using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading.Tasks;
using System.Windows.Forms;
using IrisNeuralNet.Core;
using IrisNeuralNet.Diagnostics;
using IrisNeuralNet.Lab.Experiments;
using IrisNeuralNet.Lab.Reporting;

namespace IrisNeuralNet.Lab;

/// <summary>
/// Главное окно лаборатории: композиция панели управления, графиков, видов сети и вкладок.
/// Само окно только связывает части: подготовка данных — DataPreparation, обучение — TrainingSession,
/// история графиков — RunHistory, контролы — ControlPanel.
/// </summary>
internal sealed class LabForm : Form
{
    private readonly RunHistory _history = new();
    private readonly Label _status = new();
    private readonly ControlPanel _controls = new();
    private readonly ChartBox _lossChart = new();
    private readonly ChartBox _metricChart = new();
    private readonly TopologyView _topology = new();
    private readonly DecisionBoundaryView _boundary = new();
    private readonly DigitPanel _digits = new();
    private readonly TabControl _tabs = new();
    private readonly TabPage _trainingPage = CreatePage("Обучение");
    private readonly TabPage _structurePage = CreatePage("Сеть и границы");
    private readonly TabPage _sweepPage = CreatePage("Sweep");
    private readonly TabPage _digitsPage = CreatePage("Цифры (MNIST)");
    private readonly Timer _timer = new();

    private TrainingSession? _session;
    private SweepPanel? _sweepPanel;
    private float[] _probeSample = Array.Empty<float>();
    private int _rebuildVersion;

    public LabForm()
    {
        Text = "Iris Neural Lab";
        ClientSize = new Size(1180, 760);
        MinimumSize = new Size(960, 620);
        StartPosition = FormStartPosition.CenterScreen;
        BackColor = UiTheme.Background;
        Font = UiTheme.UiFont;
        SetStyle(ControlStyles.OptimizedDoubleBuffer | ControlStyles.AllPaintingInWmPaint, true);

        BuildLayout();
        WireEvents();

        _controls.SelectDataset(0);
        _timer.Interval = 50;
        _timer.Start();
    }

    private TrainingTask? CurrentTask => _session?.Data.Task;

    private void BuildLayout()
    {
        _status.Dock = DockStyle.Top;
        _status.Height = 44;
        _status.ForeColor = UiTheme.AccentGreen;
        _status.BackColor = UiTheme.PanelBackground;
        _status.Font = UiTheme.MonoFont;
        _status.TextAlign = ContentAlignment.MiddleLeft;
        _status.Padding = new Padding(10, 0, 0, 0);

        _controls.Dock = DockStyle.Left;
        _controls.Width = 290;

        var charts = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 2, Padding = new Padding(8, 6, 8, 8), BackColor = UiTheme.Background };
        charts.RowStyles.Add(new RowStyle(SizeType.Percent, 50f));
        charts.RowStyles.Add(new RowStyle(SizeType.Percent, 50f));
        _lossChart.Dock = DockStyle.Fill;
        _lossChart.Margin = new Padding(0, 0, 0, 4);
        _metricChart.Dock = DockStyle.Fill;
        _metricChart.Margin = new Padding(0, 4, 0, 0);
        charts.Controls.Add(_lossChart, 0, 0);
        charts.Controls.Add(_metricChart, 0, 1);
        _trainingPage.Controls.Add(charts);

        var structure = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 1, Padding = new Padding(8), BackColor = UiTheme.Background };
        structure.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50f));
        structure.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50f));
        _topology.Dock = DockStyle.Fill;
        _topology.Margin = new Padding(0, 0, 4, 0);
        _boundary.Dock = DockStyle.Fill;
        _boundary.Margin = new Padding(4, 0, 0, 0);
        structure.Controls.Add(_topology, 0, 0);
        structure.Controls.Add(_boundary, 1, 0);
        _structurePage.Controls.Add(structure);

        _digitsPage.Controls.Add(_digits);

        _tabs.Dock = DockStyle.Fill;
        _tabs.TabPages.AddRange(new[] { _trainingPage, _structurePage, _sweepPage, _digitsPage });

        Controls.Add(_tabs);
        Controls.Add(_controls);
        Controls.Add(_status);
    }

    private void WireEvents()
    {
        _controls.DatasetChanged += async (_, _) => await RebuildAsync();
        _controls.DataOptionsChanged += async (_, _) => await RebuildAsync();
        _controls.LearningRateChanged += (_, _) =>
        {
            if (_session is not null) _session.LearningRate = _controls.LearningRate;
        };
        _controls.StartRequested += (_, _) => OnStart();
        _controls.StopRequested += (_, _) => _session?.Stop();
        _lossChart.Paint += (_, e) => PaintChart(e.Graphics, _lossChart.ClientRectangle, "Loss", isLoss: true);
        _metricChart.Paint += (_, e) => PaintChart(e.Graphics, _metricChart.ClientRectangle, MetricTitle(), isLoss: false);
        _timer.Tick += (_, _) => Poll();
        FormClosed += (_, _) =>
        {
            _timer.Stop();
            _session?.Dispose();
        };
    }

    /// <summary>
    /// Готовит данные выбранного датасета в фоне (MNIST — секунды). Номер версии отбрасывает
    /// устаревшие результаты, если пользователь успел переключить датасет.
    /// </summary>
    private async Task RebuildAsync()
    {
        int version = ++_rebuildVersion;
        DatasetSpec spec = _controls.SelectedSpec;
        FeatureScaling scaling = _controls.Scaling;
        float validationFraction = _controls.ValidationFraction;

        _session?.Dispose();
        _session = null;
        _history.Reset();

        try
        {
            if (!spec.IsAvailable())
            {
                if (spec.DownloadAsync is null)
                {
                    _status.Text = spec.MissingHint;
                    return;
                }

                _status.Text = spec.MissingHint;
                var progress = new Progress<string>(message =>
                {
                    if (version == _rebuildVersion) _status.Text = message;
                });
                await spec.DownloadAsync(progress, default);
            }

            _status.Text = $"{spec.Name}: загрузка и подготовка данных…";
            PreparedData data = await Task.Run(() => DataPreparation.Prepare(spec, scaling, validationFraction));
            if (version != _rebuildVersion) return;

            ApplyData(data);
        }
        catch (Exception e)
        {
            // Обработчик async void: исключение не должно уйти наружу и уронить окно.
            if (version == _rebuildVersion) _status.Text = $"{spec.Name}: ошибка подготовки данных — {e.Message}";
        }
    }

    private void ApplyData(PreparedData data)
    {
        _session = new TrainingSession(data, _controls.LearningRate);
        _probeSample = data.ValidationFeatures.AsSpan(0, data.FeatureCount).ToArray();
        _boundary.SetData(data);
        _digits.SetData(data);

        _sweepPanel?.Dispose();
        _sweepPanel = new SweepPanel(data, () => _controls.TryBuildConfig(out ExperimentConfig config) ? config : null);
        _sweepPage.Controls.Clear();
        _sweepPage.Controls.Add(_sweepPanel);

        _status.Text = $"{data.Spec.Name}: {data.TrainRowCount} train / {data.ValidationRowCount} val, {data.FeatureCount} признаков | " +
                       $"задача: {data.Task.Description} | признаки: {data.Scaling.DisplayName()} — жми Start";
        InvalidateViews();
    }

    private void OnStart()
    {
        if (_session is null || !_controls.TryBuildConfig(out ExperimentConfig config)) return;

        _history.StartNewRun(keepPrevious: _controls.OverlayPrevious);
        _session.LearningRate = _controls.LearningRate;
        _session.Start(config);
    }

    private void Poll()
    {
        TrainingSession? session = _session;
        if (session is null) return;

        _history.Consume(session.Ring);
        EpochMetrics latest = _history.Latest;
        Metric metric = session.Data.Task.Metric;
        string state = session.LastError is { } error ? $"ошибка: {error.Message}" : session.IsRunning ? "обучение" : "простой";
        _status.Text = $"epoch {latest.Epoch} | lr {session.LearningRate:0.0000} | train {metric.Name} {MetricFormat.Value(metric, latest.TrainMetric)} | " +
                       $"val {metric.Name} {MetricFormat.Value(metric, latest.ValidationMetric)} | val_loss {MetricFormat.Loss(latest.ValidationLoss)} | {state}";

        NeuralNetwork? network = session.CurrentNetwork;
        _topology.UpdateSource(network, _probeSample);
        _boundary.UpdateNetwork(network);
        if (_tabs.SelectedTab == _digitsPage) _digits.UpdateNetwork(network);
        InvalidateViews();
    }

    /// <summary>Перерисовываем только видимую вкладку: тепловая карта и топология не бесплатны.</summary>
    private void InvalidateViews()
    {
        if (_tabs.SelectedTab == _trainingPage)
        {
            _lossChart.Invalidate();
            _metricChart.Invalidate();
        }
        else if (_tabs.SelectedTab == _structurePage)
        {
            _topology.Invalidate();
            _boundary.Invalidate();
        }
    }

    private string MetricTitle() => CurrentTask is { } task ? MetricFormat.Label(task.Metric) : "Metric";

    private void PaintChart(Graphics g, Rectangle bounds, string title, bool isLoss)
    {
        var series = new List<ChartSeries>();
        if (_controls.OverlayPrevious && _history.HasPrevious)
        {
            RunSeries previous = _history.Previous;
            series.Add(new ChartSeries("train (prev)", UiTheme.PreviousRun, isLoss ? previous.TrainLoss : previous.TrainMetric));
            series.Add(new ChartSeries("val (prev)", UiTheme.PreviousRunDim, isLoss ? previous.ValidationLoss : previous.ValidationMetric));
        }

        RunSeries current = _history.Current;
        series.Add(new ChartSeries("train", isLoss ? UiTheme.AccentBlue : UiTheme.AccentGreen, isLoss ? current.TrainLoss : current.TrainMetric));
        series.Add(new ChartSeries("val", isLoss ? UiTheme.AccentOrange : UiTheme.AccentGold, isLoss ? current.ValidationLoss : current.ValidationMetric));
        ChartPainter.Draw(g, bounds, title, series);
    }

    private static TabPage CreatePage(string title) =>
        new(title) { BackColor = UiTheme.Background, UseVisualStyleBackColor = false };
}
