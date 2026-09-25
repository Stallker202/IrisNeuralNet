using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using IrisNeuralNet.Data;

namespace IrisNeuralNet.Lab
{
    internal sealed class SweepPanel : Panel
    {
        private readonly SweepRunner _runner;
        private readonly SweepHeatmapView _heatmap = new();
        private readonly NumericUpDown _epochs = new();
        private readonly Button _run = new();
        private readonly Label _status = new();
        private readonly Progress<SweepResult> _progress;
        private readonly List<SweepResult> _live = new();
        private readonly int[] _neuronOptions = { 2, 4, 8, 16, 32, 64 };
        private readonly int[] _layerOptions = { 0, 1, 2, 3 };
        private int _totalConfigs;

        public SweepPanel(PreparedData data)
        {
            Dock = DockStyle.Fill;
            BackColor = UiTheme.Background;
            _runner = new SweepRunner(
                data.TrainFeatures, data.TrainLabels,
                data.ValidationFeatures, data.ValidationLabels,
                IrisDataLoader.FeatureCount, data.ClassCount);
            _progress = new Progress<SweepResult>(OnResult);
            BuildLayout();
        }

        private void BuildLayout()
        {
            var top = new TableLayoutPanel
            {
                Dock = DockStyle.Top,
                Height = 46,
                ColumnCount = 4,
                Padding = new Padding(8, 8, 8, 4),
                BackColor = UiTheme.PanelBackground,
            };
            top.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            top.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            top.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            top.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f));

            var epochsLabel = new Label
            {
                Text = "эпох на прогон:",
                ForeColor = UiTheme.TextMuted,
                AutoSize = true,
                Margin = new Padding(0, 6, 4, 0),
            };

            _epochs.Minimum = 20;
            _epochs.Maximum = 2000;
            _epochs.Value = 150;
            _epochs.Increment = 10;
            _epochs.Width = 70;
            _epochs.BackColor = UiTheme.Background;
            _epochs.ForeColor = UiTheme.TextPrimary;
            _epochs.BorderStyle = BorderStyle.FixedSingle;
            _epochs.Margin = new Padding(0, 2, 12, 0);

            _run.Text = $"Запустить sweep ({_neuronOptions.Length * _layerOptions.Length} конфигов)";
            _run.FlatStyle = FlatStyle.Flat;
            _run.ForeColor = UiTheme.AccentGold;
            _run.BackColor = UiTheme.Background;
            _run.FlatAppearance.BorderColor = UiTheme.AccentGold;
            _run.Font = UiTheme.TitleFont;
            _run.Height = 32;
            _run.Width = 230;
            _run.Margin = new Padding(0, 0, 12, 0);

            _status.ForeColor = UiTheme.TextMuted;
            _status.AutoSize = true;
            _status.Margin = new Padding(0, 8, 0, 0);
            _status.Text = "ещё не запущено";

            top.Controls.Add(epochsLabel, 0, 0);
            top.Controls.Add(_epochs, 1, 0);
            top.Controls.Add(_run, 2, 0);
            top.Controls.Add(_status, 3, 0);

            _heatmap.Dock = DockStyle.Fill;
            _heatmap.Margin = new Padding(8);

            Controls.Add(_heatmap);
            Controls.Add(top);

            _run.Click += (_, _) => OnRun();
        }

        private void OnRun()
        {
            _run.Enabled = false;
            _live.Clear();
            _heatmap.UpdateData(_live, _neuronOptions, _layerOptions);
            _heatmap.Invalidate();
            _totalConfigs = _neuronOptions.Length * _layerOptions.Length;
            _status.Text = $"sweep: 0/{_totalConfigs}";

            var config = new SweepConfig
            {
                NeuronOptions = _neuronOptions,
                LayerOptions = _layerOptions,
                Epochs = (int)_epochs.Value,
            };

            Task.Run(() => _runner.Run(config, _progress))
                .ContinueWith(task =>
                {
                    _run.Enabled = true;
                    if (task.IsFaulted)
                    {
                        _status.Text = $"ошибка sweep: {task.Exception!.InnerExceptions[0].Message}";
                        return;
                    }

                    SweepResult best = task.Result.OrderByDescending(r => r.ValidationAccuracy).First();
                    _status.Text = $"лучшая: {best.HiddenLayers} скрытых × {best.Neurons} нейронов → " +
                                   $"val acc {best.ValidationAccuracy:P1}, loss {best.FinalLoss:F4}; прогонов: {task.Result.Count}";
                }, TaskScheduler.FromCurrentSynchronizationContext());
        }

        private void OnResult(SweepResult result)
        {
            _live.Add(result);
            _heatmap.UpdateData(_live, _neuronOptions, _layerOptions);
            _heatmap.Invalidate();
            _status.Text = $"sweep: {_live.Count}/{_totalConfigs}";
        }
    }
}
