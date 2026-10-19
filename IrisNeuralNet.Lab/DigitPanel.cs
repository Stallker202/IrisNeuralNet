using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Globalization;
using System.Windows.Forms;
using IrisNeuralNet.Core;
using IrisNeuralNet.Data;
using IrisNeuralNet.Lab.Experiments;

namespace IrisNeuralNet.Lab;

/// <summary>
/// Вкладка «Цифры» (пункт 10 методички MNIST — классификация пользовательского изображения):
/// нарисуй цифру мышью, загрузи картинку или возьми случайный тестовый образец — сеть,
/// которая сейчас обучается, сразу показывает вероятности классов.
/// </summary>
internal sealed class DigitPanel : Panel
{
    private const int CanvasSide = 280;
    private const float StrokeWidth = 22f;
    private const int Side = MnistDataLoader.ImageSide;

    private readonly Bitmap _drawing = new(CanvasSide, CanvasSide);
    private readonly PictureBox _canvas = new();
    private readonly PictureBox _preview = new();
    private readonly ProbabilityBars _bars = new();
    private readonly Label _verdict = new();
    private readonly Label _message = new();
    private readonly Button _clear = new();
    private readonly Button _open = new();
    private readonly Button _sample = new();
    private readonly NetworkProbe _probe = new();
    private readonly Random _random = new(1);
    private readonly float[] _pixels = new float[Side * Side];
    private Bitmap? _previewImage;
    private PreparedData? _data;
    private NeuralNetwork? _network;
    private Point? _lastPoint;
    private bool _hasInput;
    private string _truth = string.Empty;

    public DigitPanel()
    {
        Dock = DockStyle.Fill;
        BackColor = UiTheme.Background;
        BuildLayout();
        ClearCanvas();
        SetData(null);
    }

    /// <summary>Вкладка работает только для MNIST-подобных данных: 784 признака и 10 классов.</summary>
    public void SetData(PreparedData? data)
    {
        bool supported = data is not null && data.FeatureCount == Side * Side && data.ClassCount == 10;
        _data = supported ? data : null;
        _message.Text = supported
            ? "Рисуй цифру белым по чёрному (зажми левую кнопку) — предобработка как в MNIST: рамка 20×20 и центр масс в центре 28×28."
            : "Вкладка для MNIST: выбери датасет MNIST в левой панели и нажми Start.";
        foreach (Control control in new Control[] { _canvas, _clear, _open, _sample }) control.Enabled = supported;
        Classify();
    }

    /// <summary>Вызывается по таймеру формы: сеть обучается — вероятности обновляются вживую.</summary>
    public void UpdateNetwork(NeuralNetwork? network)
    {
        _network = network;
        Classify();
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _drawing.Dispose();
            _previewImage?.Dispose();
        }

        base.Dispose(disposing);
    }

    private void BuildLayout()
    {
        var layout = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 3, RowCount = 2, Padding = new Padding(12), BackColor = UiTheme.Background };
        layout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, CanvasSide + 24));
        layout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 200));
        layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f));
        layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100f));

        _message.ForeColor = UiTheme.TextMuted;
        _message.AutoSize = true;
        _message.Margin = new Padding(0, 0, 0, 10);
        layout.Controls.Add(_message, 0, 0);
        layout.SetColumnSpan(_message, 3);

        _canvas.Size = new Size(CanvasSide, CanvasSide);
        _canvas.Image = _drawing;
        _canvas.BorderStyle = BorderStyle.FixedSingle;
        _canvas.Cursor = Cursors.Cross;
        _canvas.MouseDown += (_, e) => BeginStroke(e);
        _canvas.MouseMove += (_, e) => ContinueStroke(e);
        _canvas.MouseUp += (_, _) => _lastPoint = null;

        StyleButton(_clear, "Очистить");
        StyleButton(_open, "Открыть изображение…");
        StyleButton(_sample, "Случайная из теста");
        _clear.Click += (_, _) => ClearCanvas();
        _open.Click += (_, _) => OpenImage();
        _sample.Click += (_, _) => ShowRandomSample();

        var left = new FlowLayoutPanel { FlowDirection = FlowDirection.TopDown, AutoSize = true, WrapContents = false, BackColor = UiTheme.Background };
        left.Controls.AddRange(new Control[] { _canvas, _clear, _open, _sample });
        layout.Controls.Add(left, 0, 1);

        var previewCaption = new Label { Text = "вход сети 28×28:", ForeColor = UiTheme.TextMuted, AutoSize = true };
        _preview.Size = new Size(168, 168);
        _preview.BorderStyle = BorderStyle.FixedSingle;
        _preview.BackColor = Color.Black;
        _preview.Paint += (_, e) => PaintPreview(e.Graphics);
        _verdict.ForeColor = UiTheme.AccentGreen;
        _verdict.Font = new Font("Segoe UI", 20f, FontStyle.Bold);
        _verdict.AutoSize = true;
        _verdict.Margin = new Padding(0, 12, 0, 0);
        var middle = new FlowLayoutPanel { FlowDirection = FlowDirection.TopDown, AutoSize = true, WrapContents = false, BackColor = UiTheme.Background };
        middle.Controls.AddRange(new Control[] { previewCaption, _preview, _verdict });
        layout.Controls.Add(middle, 1, 1);

        _bars.Dock = DockStyle.Fill;
        layout.Controls.Add(_bars, 2, 1);
        Controls.Add(layout);
    }

    private void BeginStroke(MouseEventArgs e)
    {
        if (e.Button != MouseButtons.Left) return;
        _truth = string.Empty;
        _lastPoint = e.Location;
        DrawStroke(e.Location, e.Location);
    }

    private void ContinueStroke(MouseEventArgs e)
    {
        if (e.Button != MouseButtons.Left || _lastPoint is not { } last) return;
        DrawStroke(last, e.Location);
        _lastPoint = e.Location;
    }

    private void DrawStroke(Point from, Point to)
    {
        using (Graphics g = Graphics.FromImage(_drawing))
        using (var pen = new Pen(Color.White, StrokeWidth) { StartCap = LineCap.Round, EndCap = LineCap.Round })
        {
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.DrawLine(pen, from, to.X == from.X && to.Y == from.Y ? new Point(to.X + 1, to.Y) : to);
        }

        _canvas.Invalidate();
        SetInput(BitmapPixels.ToGrayscale(_drawing), CanvasSide, CanvasSide);
    }

    private void ClearCanvas()
    {
        using (Graphics g = Graphics.FromImage(_drawing)) g.Clear(Color.Black);
        _canvas.Invalidate();
        Array.Clear(_pixels);
        _hasInput = false;
        _truth = string.Empty;
        UpdatePreview();
        Classify();
    }

    private void OpenImage()
    {
        using var dialog = new OpenFileDialog { Filter = "Изображения|*.png;*.jpg;*.jpeg;*.bmp;*.gif|Все файлы|*.*" };
        if (dialog.ShowDialog(this) != DialogResult.OK) return;

        try
        {
            using var image = new Bitmap(dialog.FileName);
            float[] gray = BitmapPixels.ToGrayscale(image);
            DigitImagePreprocessor.ToInk(gray, image.Width, image.Height);
            ShowOnCanvas(DigitImagePreprocessor.ToMnist(gray, image.Width, image.Height));
            _truth = $"файл: {System.IO.Path.GetFileName(dialog.FileName)}";
            Classify();
        }
        catch (Exception e) when (e is ArgumentException or System.IO.IOException or OutOfMemoryException)
        {
            _message.Text = $"Не удалось открыть изображение: {e.Message}";
        }
    }

    private void ShowRandomSample()
    {
        if (_data is null) return;

        Dataset test = _data.RawValidation;
        int index = _random.Next(test.RowCount);
        ShowOnCanvas(test.Row(index).ToArray());
        _truth = $"тест #{index}, истина: {test.Labels[index].ToString(CultureInfo.InvariantCulture)}";
        Classify();
    }

    /// <summary>Готовое изображение 28×28 становится входом и рисуется на холсте крупно — поверх можно дорисовать.</summary>
    private void ShowOnCanvas(float[] mnistPixels)
    {
        mnistPixels.CopyTo(_pixels, 0);
        _hasInput = true;
        using (Bitmap small = BitmapPixels.FromMnist(mnistPixels, inverted: false))
        using (Graphics g = Graphics.FromImage(_drawing))
        {
            g.InterpolationMode = InterpolationMode.NearestNeighbor;
            g.PixelOffsetMode = PixelOffsetMode.Half;
            g.DrawImage(small, new Rectangle(0, 0, CanvasSide, CanvasSide));
        }

        _canvas.Invalidate();
        UpdatePreview();
    }

    private void SetInput(float[] grayscale, int width, int height)
    {
        float[] mnist = DigitImagePreprocessor.ToMnist(grayscale, width, height);
        mnist.CopyTo(_pixels, 0);
        _hasInput = true;
        UpdatePreview();
        Classify();
    }

    private void Classify()
    {
        NeuralNetwork? network = _network;
        if (_data is null || network is null || !_hasInput || network.InputDim != _pixels.Length)
        {
            _bars.SetProbabilities(null);
            _verdict.Text = _data is not null && network is null ? "нет сети" : string.Empty;
            return;
        }

        float[] features = _data.Normalizer?.Transform(_pixels, rowCount: 1) ?? _pixels;
        float[] probabilities = _probe.Run(network, features).ToArray();
        int digit = Metrics.ArgMax(probabilities);
        _bars.SetProbabilities(probabilities);
        _verdict.Text = $"это {digit}  ({probabilities[digit].ToString("P1", CultureInfo.InvariantCulture)})"
                        + (_truth.Length > 0 ? $"\n{_truth}" : string.Empty);
    }

    private void UpdatePreview()
    {
        _previewImage?.Dispose();
        _previewImage = BitmapPixels.FromMnist(_pixels, inverted: false);
        _preview.Invalidate();
    }

    private void PaintPreview(Graphics g)
    {
        if (_previewImage is null) return;
        g.InterpolationMode = InterpolationMode.NearestNeighbor;
        g.PixelOffsetMode = PixelOffsetMode.Half;
        g.DrawImage(_previewImage, new Rectangle(0, 0, _preview.ClientSize.Width, _preview.ClientSize.Height));
    }

    private static void StyleButton(Button button, string text)
    {
        button.Text = text;
        button.FlatStyle = FlatStyle.Flat;
        button.ForeColor = UiTheme.AccentBlue;
        button.BackColor = UiTheme.Background;
        button.FlatAppearance.BorderColor = UiTheme.AccentBlue;
        button.Width = CanvasSide;
        button.Height = 30;
        button.Margin = new Padding(0, 6, 0, 0);
    }
}

/// <summary>Горизонтальные столбики вероятностей десяти цифр; максимум подсвечен.</summary>
internal sealed class ProbabilityBars : PictureBox
{
    private float[]? _probabilities;

    public ProbabilityBars()
    {
        SetStyle(ControlStyles.OptimizedDoubleBuffer | ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint, true);
        UpdateStyles();
    }

    public void SetProbabilities(float[]? probabilities)
    {
        _probabilities = probabilities;
        Invalidate();
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        Graphics g = e.Graphics;
        g.SmoothingMode = SmoothingMode.AntiAlias;
        using (var background = new SolidBrush(UiTheme.ChartBackground)) g.FillRectangle(background, ClientRectangle);

        using var title = new SolidBrush(UiTheme.TextPrimary);
        g.DrawString("Вероятности классов (softmax)", UiTheme.TitleFont, title, 8, 6);
        if (_probabilities is null) return;

        int best = Metrics.ArgMax(_probabilities);
        float rowHeight = (ClientSize.Height - 40) / (float)_probabilities.Length;
        float barLeft = 40;
        float barMax = Math.Max(10, ClientSize.Width - barLeft - 80);
        using var label = new SolidBrush(UiTheme.TextMuted);
        for (int digit = 0; digit < _probabilities.Length; digit++)
        {
            float y = 34 + digit * rowHeight;
            float p = Math.Clamp(_probabilities[digit], 0f, 1f);
            g.DrawString(digit.ToString(CultureInfo.InvariantCulture), UiTheme.TitleFont, label, 14, y + rowHeight / 2 - 8);
            using var bar = new SolidBrush(digit == best ? UiTheme.AccentGreen : UiTheme.AccentBlue);
            g.FillRectangle(bar, barLeft, y + rowHeight * 0.2f, Math.Max(1f, p * barMax), rowHeight * 0.6f);
            g.DrawString(p.ToString("P1", CultureInfo.InvariantCulture), UiTheme.MonoFont, label, barLeft + p * barMax + 6, y + rowHeight / 2 - 7);
        }
    }
}
