using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Windows.Forms;
using IrisNeuralNet.Core;
using IrisNeuralNet.Diagnostics;
using IrisNeuralNet.MathCore;

namespace IrisNeuralNet.Lab
{

    internal sealed class LabForm : Form
    {
        private readonly List<EpochMetrics> _ringSnapshot = new();
        private readonly List<float> _trainLoss = new();
        private readonly List<float> _valLoss = new();
        private readonly List<float> _trainAcc = new();
        private readonly List<float> _valAcc = new();
        private readonly List<float> _prevTrainLoss = new();
        private readonly List<float> _prevValLoss = new();
        private readonly List<float> _prevTrainAcc = new();
        private readonly List<float> _prevValAcc = new();

        private readonly Label _status = new();
        private readonly ComboBox _dataset = new();
        private readonly CheckBox _normalize = new();
        private readonly NumericUpDown _valFraction = new();
        private readonly CheckBox _earlyStop = new();
        private readonly NumericUpDown _patience = new();
        private readonly Label _lrLabel = new();
        private readonly Label _hint = new();
        private readonly TrackBar _lrSlider = new();
        private readonly NumericUpDown _neurons = new();
        private readonly NumericUpDown _hiddenLayers = new();
        private readonly NumericUpDown _epochs = new();
        private readonly NumericUpDown _batch = new();
        private readonly CheckBox _overlay = new();
        private readonly Button _start = new();
        private readonly Button _stop = new();
        private readonly ChartBox _lossChart = new();
        private readonly ChartBox _accChart = new();
        private readonly TopologyView _topology = new();
        private readonly DecisionBoundaryView _boundary = new();
        private readonly Timer _timer = new();

        private readonly TabPage _sweepPage = new() { BackColor = UiTheme.Background, UseVisualStyleBackColor = false };
        private SweepPanel? _sweepPanel;
        private TrainingSession? _session;
        private PreparedData? _data;
        private float[] _probeSample = Array.Empty<float>();
        private int _consumedEpochs;
        private bool _architectureDirty;
        private bool _suppressRebuild;

        public LabForm()
        {
            Text = "Iris Neural Lab";
            ClientSize = new Size(1120, 720);
            MinimumSize = new Size(900, 580);
            StartPosition = FormStartPosition.CenterScreen;
            BackColor = UiTheme.Background;
            Font = UiTheme.UiFont;
            SetStyle(ControlStyles.OptimizedDoubleBuffer | ControlStyles.AllPaintingInWmPaint, true);

            BuildLayout();
            WireEvents();

            _dataset.SelectedIndex = 0;
            _lrSlider.Value = 767;
            ApplyLearningRateFromSlider();
            _timer.Interval = 33;
            _timer.Start();
        }

        private DatasetSpec CurrentSpec =>
            _dataset.SelectedIndex >= 0 ? Datasets.All[_dataset.SelectedIndex] : Datasets.All[0];

        private void BuildLayout()
        {
            _status.Dock = DockStyle.Top;
            _status.Height = 44;
            _status.ForeColor = UiTheme.AccentGreen;
            _status.BackColor = UiTheme.PanelBackground;
            _status.Font = UiTheme.MonoFont;
            _status.TextAlign = ContentAlignment.MiddleLeft;
            _status.Padding = new Padding(10, 0, 0, 0);

            var left = new TableLayoutPanel
            {
                Dock = DockStyle.Left,
                Width = 272,
                ColumnCount = 1,
                Padding = new Padding(10),
                BackColor = UiTheme.PanelBackground,
            };

            _dataset.DropDownStyle = ComboBoxStyle.DropDownList;
            _dataset.Dock = DockStyle.Fill;
            _dataset.BackColor = UiTheme.Background;
            _dataset.ForeColor = UiTheme.TextPrimary;
            foreach (DatasetSpec spec in Datasets.All)
            {
                _dataset.Items.Add(spec.Name);
            }

            _normalize.Text = "нормализация (z-score)";
            _normalize.ForeColor = UiTheme.TextMuted;
            _normalize.AutoSize = true;
            _normalize.Margin = new Padding(2, 8, 2, 2);

            _valFraction.Minimum = 0.05m;
            _valFraction.Maximum = 0.5m;
            _valFraction.Increment = 0.05m;
            _valFraction.DecimalPlaces = 2;
            StyleNumericBox(_valFraction);

            _earlyStop.Text = "early stopping";
            _earlyStop.ForeColor = UiTheme.TextMuted;
            _earlyStop.AutoSize = true;
            _earlyStop.Margin = new Padding(2, 8, 2, 2);

            _patience.Minimum = 5;
            _patience.Maximum = 200;
            _patience.Value = 20;
            _patience.Increment = 5;
            StyleNumericBox(_patience);

            _lrLabel.ForeColor = UiTheme.TextPrimary;
            _lrLabel.AutoSize = true;
            _lrLabel.Margin = new Padding(2, 4, 2, 2);

            _lrSlider.Dock = DockStyle.Fill;
            _lrSlider.Minimum = 0;
            _lrSlider.Maximum = 1000;
            _lrSlider.TickFrequency = 100;
            _lrSlider.BackColor = UiTheme.PanelBackground;
            _lrSlider.Margin = new Padding(2, 6, 2, 10);

            StyleNumericBox(_neurons); _neurons.Minimum = 1; _neurons.Maximum = 128; _neurons.Value = 8;
            StyleNumericBox(_hiddenLayers); _hiddenLayers.Minimum = 0; _hiddenLayers.Maximum = 3; _hiddenLayers.Value = 1;
            StyleNumericBox(_epochs); _epochs.Minimum = 10; _epochs.Maximum = 2000; _epochs.Value = 200; _epochs.Increment = 10;
            StyleNumericBox(_batch); _batch.Minimum = 1; _batch.Maximum = 208; _batch.Value = 16;

            _overlay.Text = "оверлей прошлого прогона";
            _overlay.Checked = true;
            _overlay.ForeColor = UiTheme.TextMuted;
            _overlay.Margin = new Padding(2, 10, 2, 6);

            StyleButton(_start, "Start / Restart", UiTheme.AccentGreen);
            StyleButton(_stop, "Stop", UiTheme.AccentRed);

            _hint.ForeColor = UiTheme.AccentOrange;
            _hint.AutoSize = true;
            _hint.Margin = new Padding(2, 8, 2, 2);

            AddRow(left, SectionLabel("датасет"));
            AddRow(left, _dataset);
            AddRow(left, _normalize);
            AddRow(left, SectionLabel("validation split"));
            AddRow(left, _valFraction);
            AddRow(left, _earlyStop);
            AddRow(left, _patience);
            AddRow(left, _lrLabel);
            AddRow(left, _lrSlider);
            AddRow(left, SectionLabel("нейронов в скрытом слое"));
            AddRow(left, _neurons);
            AddRow(left, SectionLabel("скрытых слоёв"));
            AddRow(left, _hiddenLayers);
            AddRow(left, SectionLabel("эпох"));
            AddRow(left, _epochs);
            AddRow(left, SectionLabel("batch size"));
            AddRow(left, _batch);
            AddRow(left, _overlay);
            AddRow(left, _start);
            AddRow(left, _stop);
            AddRow(left, _hint);
            left.RowCount++;
            left.RowStyles.Add(new RowStyle(SizeType.Percent, 100f));

            var charts = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 2,
                Padding = new Padding(8, 6, 8, 8),
                BackColor = UiTheme.Background,
            };
            charts.RowStyles.Add(new RowStyle(SizeType.Percent, 50f));
            charts.RowStyles.Add(new RowStyle(SizeType.Percent, 50f));
            _lossChart.Dock = DockStyle.Fill;
            _lossChart.Margin = new Padding(0, 0, 0, 4);
            _accChart.Dock = DockStyle.Fill;
            _accChart.Margin = new Padding(0, 4, 0, 0);
            charts.Controls.Add(_lossChart, 0, 0);
            charts.Controls.Add(_accChart, 0, 1);

            var split = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 2,
                RowCount = 1,
                Padding = new Padding(8),
                BackColor = UiTheme.Background,
            };
            split.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50f));
            split.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50f));
            _topology.Dock = DockStyle.Fill;
            _topology.Margin = new Padding(0, 0, 4, 0);
            _boundary.Dock = DockStyle.Fill;
            _boundary.Margin = new Padding(4, 0, 0, 0);
            split.Controls.Add(_topology, 0, 0);
            split.Controls.Add(_boundary, 1, 0);

            var trainingPage = new TabPage("Обучение") { BackColor = UiTheme.Background, UseVisualStyleBackColor = false };
            var structurePage = new TabPage("Сеть и границы") { BackColor = UiTheme.Background, UseVisualStyleBackColor = false };
            trainingPage.Controls.Add(charts);
            structurePage.Controls.Add(split);

            var tabs = new TabControl { Dock = DockStyle.Fill };
            tabs.TabPages.Add(trainingPage);
            tabs.TabPages.Add(structurePage);
            tabs.TabPages.Add(_sweepPage);
            _sweepPage.Text = "Sweep";

            Controls.Add(tabs);
            Controls.Add(left);
            Controls.Add(_status);
        }

        private static void StyleNumericBox(NumericUpDown control)
        {
            control.Dock = DockStyle.Fill;
            control.Margin = new Padding(2, 2, 2, 6);
            control.BackColor = UiTheme.Background;
            control.ForeColor = UiTheme.TextPrimary;
            control.BorderStyle = BorderStyle.FixedSingle;
        }

        private static Label SectionLabel(string text) =>
            new Label
            {
                Text = text,
                ForeColor = UiTheme.TextMuted,
                AutoSize = true,
                Margin = new Padding(2, 8, 2, 2),
            };

        private static void StyleButton(Button button, string text, Color accent)
        {
            button.Text = text;
            button.FlatStyle = FlatStyle.Flat;
            button.ForeColor = accent;
            button.BackColor = UiTheme.Background;
            button.FlatAppearance.BorderColor = accent;
            button.FlatAppearance.MouseOverBackColor = Color.FromArgb(48, 52, 60);
            button.Font = UiTheme.TitleFont;
            button.Height = 34;
            button.Dock = DockStyle.Fill;
            button.Margin = new Padding(2, 6, 2, 2);
        }

        private static void AddRow(TableLayoutPanel panel, Control control)
        {
            panel.RowCount++;
            panel.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            panel.Controls.Add(control, 0, panel.RowCount - 1);
        }

        private void WireEvents()
        {
            _dataset.SelectedIndexChanged += (_, _) => SelectDataset(CurrentSpec);
            _normalize.CheckedChanged += (_, _) => RebuildIfAllowed();
            _valFraction.ValueChanged += (_, _) => RebuildIfAllowed();
            _earlyStop.CheckedChanged += (_, _) => _patience.Enabled = _earlyStop.Checked;
            _lrSlider.Scroll += (_, _) => ApplyLearningRateFromSlider();
            _neurons.ValueChanged += (_, _) => MarkArchitectureDirty();
            _hiddenLayers.ValueChanged += (_, _) => MarkArchitectureDirty();
            _start.Click += (_, _) => OnStart();
            _stop.Click += (_, _) => _session?.Stop();
            _lossChart.Paint += (_, e) => PaintLoss(e.Graphics, _lossChart.ClientRectangle);
            _accChart.Paint += (_, e) => PaintAccuracy(e.Graphics, _accChart.ClientRectangle);
            _timer.Tick += (_, _) => Poll();
            FormClosed += (_, _) =>
            {
                _timer.Stop();
                _session?.Stop();
            };
        }

        private void SelectDataset(DatasetSpec spec)
        {
            _suppressRebuild = true;
            _normalize.Checked = spec.DefaultNormalize;
            _valFraction.Value = (decimal)spec.DefaultValidationFraction;
            _neurons.Value = Math.Clamp(spec.DefaultHidden[0], (int)_neurons.Minimum, (int)_neurons.Maximum);
            _hiddenLayers.Value = Math.Clamp(spec.DefaultHidden.Length, (int)_hiddenLayers.Minimum, (int)_hiddenLayers.Maximum);
            _patience.Enabled = _earlyStop.Checked;
            _suppressRebuild = false;
            Rebuild();
        }

        private void RebuildIfAllowed()
        {
            if (!_suppressRebuild)
            {
                Rebuild();
            }
        }

        private void Rebuild()
        {
            DatasetSpec spec = CurrentSpec;
            if (!File.Exists(spec.FileName))
            {
                _status.Text = $"нет файла '{spec.FileName}' — положи его в папку Lab";
                return;
            }

            _data = DataPreparation.Prepare(spec, (float)_valFraction.Value, _normalize.Checked);
            _session?.Dispose();
            _session = new TrainingSession(_data);

            float[] allFeatures = Concat(_data.TrainFeatures, _data.ValidationFeatures);
            int[] allClasses = Concat(_data.TrainClasses, _data.ValidationClasses);
            Pca2D pca = Pca2D.Fit(allFeatures, _data.FeatureCount);
            _boundary.SetData(pca, allFeatures, allClasses, _data.FeatureCount, _data.ClassCount);

            _probeSample = new float[_data.FeatureCount];
            Array.Copy(_data.ValidationFeatures, _probeSample, _data.FeatureCount);

            _sweepPanel = new SweepPanel(_data) { Dock = DockStyle.Fill };
            _sweepPage.Controls.Clear();
            _sweepPage.Controls.Add(_sweepPanel);

            _trainLoss.Clear(); _valLoss.Clear(); _trainAcc.Clear(); _valAcc.Clear();
            _prevTrainLoss.Clear(); _prevValLoss.Clear(); _prevTrainAcc.Clear(); _prevValAcc.Clear();
            _consumedEpochs = 0;
            _architectureDirty = false;
            UpdateHint();

            _status.Text = $"{spec.Name}: {_data.TrainRowCount} train / {_data.ValidationRowCount} val | " +
                           $"задача: {(spec.TaskKind == TaskKind.Binary ? "бинарная (sigmoid+BCE)" : "multi-class (softmax+CE)")}";
        }

        private static float[] Concat(float[] a, float[] b)
        {
            var result = new float[a.Length + b.Length];
            a.CopyTo(result, 0);
            b.CopyTo(result, a.Length);
            return result;
        }

        private static int[] Concat(int[] a, int[] b)
        {
            var result = new int[a.Length + b.Length];
            a.CopyTo(result, 0);
            b.CopyTo(result, a.Length);
            return result;
        }

        private void ApplyLearningRateFromSlider()
        {
            if (_session is not null)
            {
                _session.LearningRate = SliderToLearningRate(_lrSlider.Value);
            }

            _lrLabel.Text = $"learning rate: {SliderToLearningRate(_lrSlider.Value):F4} (live)";
        }

        private static float SliderToLearningRate(int value) =>
            (float)(1e-4 * Math.Pow(10, value / 1000.0 * 3.0));

        private void MarkArchitectureDirty()
        {
            _architectureDirty = true;
            UpdateHint();
        }

        private void UpdateHint() =>
            _hint.Text = _architectureDirty ? "архитектура изменена — жми Start/Restart" : string.Empty;

        private int[] BuildArchitecture()
        {
            int layers = (int)_hiddenLayers.Value;
            int neurons = (int)_neurons.Value;
            var architecture = new int[layers];
            for (int i = 0; i < layers; i++)
            {
                architecture[i] = neurons;
            }

            return architecture;
        }

        private void OnStart()
        {
            if (_session is null)
            {
                return;
            }

            if (_overlay.Checked && _trainLoss.Count > 1)
            {
                CopyList(_trainLoss, _prevTrainLoss);
                CopyList(_valLoss, _prevValLoss);
                CopyList(_trainAcc, _prevTrainAcc);
                CopyList(_valAcc, _prevValAcc);
            }

            _trainLoss.Clear(); _valLoss.Clear(); _trainAcc.Clear(); _valAcc.Clear();
            _consumedEpochs = 0;
            _architectureDirty = false;
            UpdateHint();

            int patience = _earlyStop.Checked ? (int)_patience.Value : 0;
            _session.Start(BuildArchitecture(), (int)_epochs.Value, (int)_batch.Value, seed: 7, earlyStoppingPatience: patience);
        }

        private static void CopyList(List<float> from, List<float> to)
        {
            to.Clear();
            to.AddRange(from);
        }

        private void Poll()
        {
            if (_session is null)
            {
                return;
            }

            _session.Ring.CopyTo(_ringSnapshot);
            if (_ringSnapshot.Count > _consumedEpochs)
            {
                for (int i = _consumedEpochs; i < _ringSnapshot.Count; i++)
                {
                    EpochMetrics m = _ringSnapshot[i];
                    _trainLoss.Add(m.TrainLoss);
                    _valLoss.Add(m.ValidationLoss);
                    _trainAcc.Add(m.TrainAccuracy);
                    _valAcc.Add(m.ValidationAccuracy);
                }

                _consumedEpochs = _ringSnapshot.Count;
            }

            EpochMetrics latest = _ringSnapshot.Count > 0 ? _ringSnapshot[_ringSnapshot.Count - 1] : default;
            _status.Text = $"epoch {latest.Epoch} | lr {_session.LearningRate:F4} | train acc {latest.TrainAccuracy:P1} | " +
                           $"val acc {latest.ValidationAccuracy:P1} | {(_session.IsRunning ? "обучение" : "простой")}";

            _lossChart.Invalidate();
            _accChart.Invalidate();
            _topology.UpdateSource(_session.CurrentNetwork, _probeSample);
            _topology.Invalidate();
            _boundary.UpdateNetwork(_session.CurrentNetwork);
            _boundary.Invalidate();
        }

        private void PaintLoss(Graphics g, Rectangle bounds)
        {
            var series = new List<ChartSeries>();
            if (_overlay.Checked && _prevTrainLoss.Count > 1)
            {
                series.Add(new ChartSeries("train (prev)", UiTheme.PreviousRun, _prevTrainLoss));
                series.Add(new ChartSeries("val (prev)", UiTheme.PreviousRunDim, _prevValLoss));
            }

            series.Add(new ChartSeries("train", UiTheme.AccentBlue, _trainLoss));
            series.Add(new ChartSeries("val", UiTheme.AccentOrange, _valLoss));
            ChartPainter.Draw(g, bounds, "Loss", series);
        }

        private void PaintAccuracy(Graphics g, Rectangle bounds)
        {
            var series = new List<ChartSeries>();
            if (_overlay.Checked && _prevTrainAcc.Count > 1)
            {
                series.Add(new ChartSeries("train (prev)", UiTheme.PreviousRun, _prevTrainAcc));
                series.Add(new ChartSeries("val (prev)", UiTheme.PreviousRunDim, _prevTrainAcc.Count > 1 ? _prevValAcc : _prevValAcc));
            }

            series.Add(new ChartSeries("train", UiTheme.AccentGreen, _trainAcc));
            series.Add(new ChartSeries("val", UiTheme.AccentGold, _valAcc));
            ChartPainter.Draw(g, bounds, "Accuracy", series);
        }
    }
}