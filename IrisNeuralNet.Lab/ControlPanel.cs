using System;
using System.Drawing;
using System.Globalization;
using System.Windows.Forms;
using IrisNeuralNet.Lab.Experiments;
using IrisNeuralNet.Optimizers;

namespace IrisNeuralNet.Lab;

/// <summary>
/// Левая панель лаборатории: датасет, предобработка, архитектура и гиперпараметры.
/// Владеет только контролами и их валидацией; что делать по событиям, решает LabForm.
/// </summary>
internal sealed class ControlPanel : TableLayoutPanel
{
    private const int SliderMax = 1000;
    private const double MinLearningRateLog = -4.0;
    private const double LearningRateDecades = 3.0;

    private static readonly OptimizerKind[] Optimizers =
        { OptimizerKind.Adam, OptimizerKind.RmsProp, OptimizerKind.Sgd, OptimizerKind.SgdMomentum };

    private readonly ComboBox _dataset = new();
    private readonly ComboBox _scaling = new();
    private readonly NumericUpDown _validationFraction = new();
    private readonly TextBox _architecture = new();
    private readonly Label _architectureError = new();
    private readonly ComboBox _optimizer = new();
    private readonly Label _learningRateLabel = new();
    private readonly TrackBar _learningRate = new();
    private readonly NumericUpDown _epochs = new();
    private readonly NumericUpDown _batch = new();
    private readonly CheckBox _earlyStop = new();
    private readonly NumericUpDown _patience = new();
    private readonly CheckBox _overlay = new();
    private readonly Button _start = new();
    private readonly Button _stop = new();
    private readonly Label _hint = new();
    private bool _suppressEvents;

    public ControlPanel()
    {
        ColumnCount = 1;
        Padding = new Padding(10);
        BackColor = UiTheme.PanelBackground;
        AutoScroll = true;
        BuildControls();
        WireEvents();
    }

    /// <summary>Выбран другой датасет; значения по умолчанию уже применены к контролам.</summary>
    public event EventHandler? DatasetChanged;

    /// <summary>Изменилась предобработка или доля валидации — данные нужно подготовить заново.</summary>
    public event EventHandler? DataOptionsChanged;

    public event EventHandler? LearningRateChanged;
    public event EventHandler? StartRequested;
    public event EventHandler? StopRequested;

    public DatasetSpec SelectedSpec => Datasets.All[Math.Max(0, _dataset.SelectedIndex)];
    public FeatureScaling Scaling => (FeatureScaling)Math.Max(0, _scaling.SelectedIndex);
    public float ValidationFraction => (float)_validationFraction.Value;
    public float LearningRate => SliderToLearningRate(_learningRate.Value);
    public bool OverlayPrevious => _overlay.Checked;

    public void SelectDataset(int index) => _dataset.SelectedIndex = index;

    public void SetHint(string text) => _hint.Text = text;

    /// <summary>Конфиг прогона из контролов; false и подсказка, если архитектура записана с ошибкой.</summary>
    public bool TryBuildConfig(out ExperimentConfig config)
    {
        config = null!;
        if (!ArchitectureParser.TryParse(_architecture.Text, out int[] hidden, out string error))
        {
            _architectureError.Text = error;
            return false;
        }

        config = new ExperimentConfig(
            Name: $"{SelectedSpec.Name} [{ArchitectureParser.Format(hidden)}]",
            Hidden: hidden,
            Optimizer: Optimizers[Math.Max(0, _optimizer.SelectedIndex)],
            LearningRate: LearningRate,
            Epochs: (int)_epochs.Value,
            BatchSize: (int)_batch.Value,
            Seed: 7,
            EarlyStoppingPatience: _earlyStop.Checked ? (int)_patience.Value : 0);
        return true;
    }

    private void ApplyDefaults(DatasetSpec spec)
    {
        _suppressEvents = true;
        try
        {
            ExperimentDefaults defaults = spec.Defaults;
            _scaling.SelectedIndex = (int)defaults.Scaling;
            _validationFraction.Enabled = !spec.HasPredefinedTestSplit;
            if (!spec.HasPredefinedTestSplit)
            {
                _validationFraction.Value = (decimal)defaults.ValidationFraction;
            }

            _architecture.Text = ArchitectureParser.Format(defaults.Hidden);
            _architectureError.Text = string.Empty;
            _optimizer.SelectedIndex = Array.IndexOf(Optimizers, defaults.Optimizer);
            _learningRate.Value = LearningRateToSlider(defaults.LearningRate);
            _epochs.Value = Math.Clamp(defaults.Epochs, (int)_epochs.Minimum, (int)_epochs.Maximum);
            _batch.Value = Math.Clamp(defaults.BatchSize, (int)_batch.Minimum, (int)_batch.Maximum);
            UpdateLearningRateLabel();
            _hint.Text = string.Empty;
        }
        finally
        {
            _suppressEvents = false;
        }
    }

    private void BuildControls()
    {
        StyleCombo(_dataset);
        foreach (DatasetSpec spec in Datasets.All) _dataset.Items.Add(spec.Name);

        StyleCombo(_scaling);
        foreach (FeatureScaling scaling in Enum.GetValues<FeatureScaling>()) _scaling.Items.Add(scaling.DisplayName());

        StyleNumeric(_validationFraction, 0.05m, 0.5m, 0.2m, 0.05m);
        _validationFraction.DecimalPlaces = 2;

        _architecture.BackColor = UiTheme.Background;
        _architecture.ForeColor = UiTheme.TextPrimary;
        _architecture.BorderStyle = BorderStyle.FixedSingle;
        _architecture.Font = UiTheme.MonoFont;
        _architecture.Dock = DockStyle.Fill;
        _architecture.Margin = new Padding(2, 2, 2, 2);

        _architectureError.ForeColor = UiTheme.AccentRed;
        _architectureError.AutoSize = true;
        _architectureError.Margin = new Padding(2, 0, 2, 4);

        StyleCombo(_optimizer);
        foreach (OptimizerKind kind in Optimizers) _optimizer.Items.Add(OptimizerFactory.KerasName(kind));

        _learningRateLabel.ForeColor = UiTheme.TextPrimary;
        _learningRateLabel.AutoSize = true;
        _learningRateLabel.Margin = new Padding(2, 8, 2, 2);

        _learningRate.Dock = DockStyle.Fill;
        _learningRate.Minimum = 0;
        _learningRate.Maximum = SliderMax;
        _learningRate.TickFrequency = SliderMax / 6;
        _learningRate.BackColor = UiTheme.PanelBackground;
        _learningRate.Margin = new Padding(2, 2, 2, 6);

        StyleNumeric(_epochs, 1, 5000, 200, 10);
        StyleNumeric(_batch, 1, 1024, 16, 1);
        StyleCheck(_earlyStop, "early stopping (patience, эпох)");
        StyleNumeric(_patience, 1, 500, 20, 5);
        _patience.Enabled = false;
        StyleCheck(_overlay, "оверлей прошлого прогона");
        _overlay.Checked = true;

        StyleButton(_start, "Start / Restart", UiTheme.AccentGreen);
        StyleButton(_stop, "Stop", UiTheme.AccentRed);

        _hint.ForeColor = UiTheme.AccentOrange;
        _hint.AutoSize = true;
        _hint.MaximumSize = new Size(250, 0);
        _hint.Margin = new Padding(2, 8, 2, 2);

        AddRow(SectionLabel("датасет"));
        AddRow(_dataset);
        AddRow(SectionLabel("предобработка признаков"));
        AddRow(_scaling);
        AddRow(SectionLabel("доля validation"));
        AddRow(_validationFraction);
        AddRow(SectionLabel("скрытые слои (напр. 512, 256)"));
        AddRow(_architecture);
        AddRow(_architectureError);
        AddRow(SectionLabel("оптимизатор"));
        AddRow(_optimizer);
        AddRow(_learningRateLabel);
        AddRow(_learningRate);
        AddRow(SectionLabel("эпох"));
        AddRow(_epochs);
        AddRow(SectionLabel("batch size"));
        AddRow(_batch);
        AddRow(_earlyStop);
        AddRow(_patience);
        AddRow(_overlay);
        AddRow(_start);
        AddRow(_stop);
        AddRow(_hint);
        RowCount++;
        RowStyles.Add(new RowStyle(SizeType.Percent, 100f));
    }

    private void WireEvents()
    {
        _dataset.SelectedIndexChanged += (_, _) =>
        {
            ApplyDefaults(SelectedSpec);
            DatasetChanged?.Invoke(this, EventArgs.Empty);
        };
        _scaling.SelectedIndexChanged += (_, _) => RaiseUnlessSuppressed(DataOptionsChanged);
        _validationFraction.ValueChanged += (_, _) => RaiseUnlessSuppressed(DataOptionsChanged);
        _architecture.TextChanged += (_, _) => OnArchitectureEdited();
        _optimizer.SelectedIndexChanged += (_, _) => MarkChangedUnlessSuppressed("оптимизатор изменён — жми Start/Restart");
        _learningRate.Scroll += (_, _) =>
        {
            UpdateLearningRateLabel();
            LearningRateChanged?.Invoke(this, EventArgs.Empty);
        };
        _earlyStop.CheckedChanged += (_, _) => _patience.Enabled = _earlyStop.Checked;
        _start.Click += (_, _) =>
        {
            _hint.Text = string.Empty;
            StartRequested?.Invoke(this, EventArgs.Empty);
        };
        _stop.Click += (_, _) => StopRequested?.Invoke(this, EventArgs.Empty);
    }

    private void OnArchitectureEdited()
    {
        _architectureError.Text = ArchitectureParser.TryParse(_architecture.Text, out _, out string error) ? string.Empty : error;
        MarkChangedUnlessSuppressed("архитектура изменена — жми Start/Restart");
    }

    private void MarkChangedUnlessSuppressed(string hint)
    {
        if (!_suppressEvents) _hint.Text = hint;
    }

    private void RaiseUnlessSuppressed(EventHandler? handler)
    {
        if (!_suppressEvents) handler?.Invoke(this, EventArgs.Empty);
    }

    private void UpdateLearningRateLabel() =>
        _learningRateLabel.Text = $"learning rate: {LearningRate.ToString("0.0000", CultureInfo.InvariantCulture)} (live)";

    private static float SliderToLearningRate(int value) =>
        (float)Math.Pow(10, MinLearningRateLog + value / (double)SliderMax * LearningRateDecades);

    private static int LearningRateToSlider(float learningRate) =>
        Math.Clamp((int)Math.Round((Math.Log10(learningRate) - MinLearningRateLog) / LearningRateDecades * SliderMax), 0, SliderMax);

    private void AddRow(Control control)
    {
        RowCount++;
        RowStyles.Add(new RowStyle(SizeType.AutoSize));
        Controls.Add(control, 0, RowCount - 1);
    }

    private static Label SectionLabel(string text) =>
        new()
        {
            Text = text,
            ForeColor = UiTheme.TextMuted,
            AutoSize = true,
            Margin = new Padding(2, 8, 2, 2),
        };

    private static void StyleCombo(ComboBox combo)
    {
        combo.DropDownStyle = ComboBoxStyle.DropDownList;
        combo.Dock = DockStyle.Fill;
        combo.BackColor = UiTheme.Background;
        combo.ForeColor = UiTheme.TextPrimary;
        combo.FlatStyle = FlatStyle.Flat;
    }

    private static void StyleNumeric(NumericUpDown control, decimal minimum, decimal maximum, decimal value, decimal increment)
    {
        control.Minimum = minimum;
        control.Maximum = maximum;
        control.Value = value;
        control.Increment = increment;
        control.Dock = DockStyle.Fill;
        control.Margin = new Padding(2, 2, 2, 6);
        control.BackColor = UiTheme.Background;
        control.ForeColor = UiTheme.TextPrimary;
        control.BorderStyle = BorderStyle.FixedSingle;
    }

    private static void StyleCheck(CheckBox check, string text)
    {
        check.Text = text;
        check.ForeColor = UiTheme.TextMuted;
        check.AutoSize = true;
        check.Margin = new Padding(2, 8, 2, 2);
    }

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
}
