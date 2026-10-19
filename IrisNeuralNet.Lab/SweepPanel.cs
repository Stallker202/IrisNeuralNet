using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using IrisNeuralNet.Lab.Experiments;
using IrisNeuralNet.Lab.Reporting;

namespace IrisNeuralNet.Lab;

/// <summary>Вкладка Sweep: запуск сетки архитектур с текущими гиперпараметрами панели управления.</summary>
internal sealed class SweepPanel : Panel
{
    private static readonly SweepGrid Grid = new(LayerOptions: new[] { 0, 1, 2, 3 }, NeuronOptions: new[] { 2, 4, 8, 16, 32, 64 });

    private readonly SweepRunner _runner;
    private readonly Func<ExperimentConfig?> _template;
    private readonly SweepHeatmapView _heatmap = new();
    private readonly NumericUpDown _epochs = new();
    private readonly Button _run = new();
    private readonly Button _cancel = new();
    private readonly Label _status = new();
    private readonly List<SweepResult> _live = new();
    private CancellationTokenSource? _cancellation;

    /// <param name="template">Текущие настройки панели (оптимизатор, lr, batch); null — настройки с ошибкой.</param>
    public SweepPanel(PreparedData data, Func<ExperimentConfig?> template)
    {
        Dock = DockStyle.Fill;
        BackColor = UiTheme.Background;
        _runner = new SweepRunner(data);
        _template = template;
        BuildLayout(data.Spec.Defaults.Epochs);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing) _cancellation?.Cancel();
        base.Dispose(disposing);
    }

    private void BuildLayout(int defaultEpochs)
    {
        var top = new FlowLayoutPanel
        {
            Dock = DockStyle.Top,
            Height = 46,
            Padding = new Padding(8, 8, 8, 4),
            BackColor = UiTheme.PanelBackground,
            WrapContents = false,
        };

        var epochsLabel = new Label { Text = "эпох на прогон:", ForeColor = UiTheme.TextMuted, AutoSize = true, Margin = new Padding(0, 8, 4, 0) };
        _epochs.Minimum = 1;
        _epochs.Maximum = 5000;
        _epochs.Value = Math.Clamp(defaultEpochs, 1, 5000);
        _epochs.Width = 70;
        _epochs.BackColor = UiTheme.Background;
        _epochs.ForeColor = UiTheme.TextPrimary;
        _epochs.BorderStyle = BorderStyle.FixedSingle;
        _epochs.Margin = new Padding(0, 4, 12, 0);

        StyleButton(_run, $"Запустить sweep ({Grid.Count} конфигов)", UiTheme.AccentGold, 240);
        StyleButton(_cancel, "Отмена", UiTheme.AccentRed, 90);
        _cancel.Enabled = false;

        _status.ForeColor = UiTheme.TextMuted;
        _status.AutoSize = true;
        _status.Margin = new Padding(0, 8, 0, 0);
        _status.Text = "ещё не запущено (оптимизатор, lr и batch — из левой панели)";

        top.Controls.AddRange(new Control[] { epochsLabel, _epochs, _run, _cancel, _status });
        _heatmap.Dock = DockStyle.Fill;
        Controls.Add(_heatmap);
        Controls.Add(top);

        _run.Click += async (_, _) => await RunAsync();
        _cancel.Click += (_, _) => _cancellation?.Cancel();
    }

    private async Task RunAsync()
    {
        ExperimentConfig? template = _template();
        if (template is null)
        {
            _status.Text = "исправь архитектуру/настройки в левой панели";
            return;
        }

        template = template with { Epochs = (int)_epochs.Value };
        _run.Enabled = false;
        _cancel.Enabled = true;
        _live.Clear();
        _heatmap.UpdateData(_live, Grid, _runner.Metric);
        _heatmap.Invalidate();
        _status.Text = $"sweep: 0/{Grid.Count}";

        _cancellation = new CancellationTokenSource();
        var progress = new Progress<SweepResult>(OnResult);
        try
        {
            CancellationToken token = _cancellation.Token;
            List<SweepResult> results = await Task.Run(() => _runner.Run(Grid, template, progress, token));
            SweepResult best = SweepRunner.Best(results, _runner.Metric);
            _status.Text = $"лучшая: {best.HiddenLayers} скрытых × {best.Neurons} нейронов → val {_runner.Metric.Name} " +
                           $"{MetricFormat.Value(_runner.Metric, best.ValidationMetric)}, val_loss {MetricFormat.Loss(best.ValidationLoss)}; прогонов: {results.Count}";
        }
        catch (Exception e) when (IsCancellation(e))
        {
            _status.Text = $"sweep отменён: готово {_live.Count}/{Grid.Count}";
        }
        catch (Exception e)
        {
            _status.Text = $"ошибка sweep: {e.Message}";
        }
        finally
        {
            _run.Enabled = true;
            _cancel.Enabled = false;
            _cancellation.Dispose();
            _cancellation = null;
        }
    }

    /// <summary>Parallel.ForEach заворачивает отмену из тела цикла в AggregateException.</summary>
    private static bool IsCancellation(Exception e) =>
        e is OperationCanceledException
        || e is AggregateException aggregate && aggregate.Flatten().InnerExceptions.All(inner => inner is OperationCanceledException);

    private void OnResult(SweepResult result)
    {
        _live.Add(result);
        _heatmap.UpdateData(_live, Grid, _runner.Metric);
        _heatmap.Invalidate();
        _status.Text = $"sweep: {_live.Count}/{Grid.Count}";
    }

    private static void StyleButton(Button button, string text, Color accent, int width)
    {
        button.Text = text;
        button.FlatStyle = FlatStyle.Flat;
        button.ForeColor = accent;
        button.BackColor = UiTheme.Background;
        button.FlatAppearance.BorderColor = accent;
        button.Font = UiTheme.TitleFont;
        button.Height = 32;
        button.Width = width;
        button.Margin = new Padding(0, 0, 12, 0);
    }
}
