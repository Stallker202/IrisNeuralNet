using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;
using IrisNeuralNet.Core;
using IrisNeuralNet.MathCore;

namespace IrisNeuralNet.Lab
{
    internal sealed class DecisionBoundaryView : PictureBox
    {
        private const int GridW = 56;
        private const int GridH = 40;

        private static readonly Color[] ClassColors =
        {
        UiTheme.AccentBlue, UiTheme.AccentGreen, UiTheme.AccentOrange, UiTheme.AccentGold, UiTheme.AccentRed,
    };

        private readonly NetworkProbe _probe = new();
        private Pca2D? _pca;
        private NeuralNetwork? _network;
        private float[] _probeBuffer = Array.Empty<float>();
        private float[] _features = Array.Empty<float>();
        private int[] _classes = Array.Empty<int>();
        private float[] _projected = Array.Empty<float>();
        private int _dim;
        private SolidBrush?[,]? _cellBrushes;

        public DecisionBoundaryView()
        {
            SetStyle(ControlStyles.OptimizedDoubleBuffer | ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint, true);
            UpdateStyles();
        }

        public void SetData(Pca2D pca, float[] features, int[] classes, int dim, int classCount)
        {
            _pca = pca;
            _features = features;
            _classes = classes;
            _dim = dim;
            _probeBuffer = new float[dim];
            _projected = new float[classes.Length * 2];
            for (int i = 0; i < classes.Length; i++)
            {
                pca.Project(features.AsSpan(i * dim, dim), out _projected[i * 2], out _projected[i * 2 + 1]);
            }
        }

        public void UpdateNetwork(NeuralNetwork? network) => _network = network;

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            Graphics g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            using (var bg = new SolidBrush(UiTheme.ChartBackground))
            {
                g.FillRectangle(bg, ClientRectangle);
            }

            if (_pca is null || _classes.Length == 0)
            {
                using var waiting = new SolidBrush(UiTheme.TextMuted);
                g.DrawString("ожидание данные...", UiTheme.UiFont, waiting, 10, 10);
                return;
            }

            using var titleBrush = new SolidBrush(UiTheme.TextPrimary);
            g.DrawString("Границы решений: PCA 2D", UiTheme.TitleFont, titleBrush, 8, 6);

            Rectangle plot = Rectangle.FromLTRB(40, 30, ClientRectangle.Right - 12, ClientRectangle.Bottom - 30);
            if (plot.Width < 20 || plot.Height < 20)
            {
                return;
            }

            NeuralNetwork? network = _network;
            if (network is not null && network.InputDim == _dim)
            {
                float cellW = plot.Width / (float)GridW;
                float cellH = plot.Height / (float)GridH;
                for (int gy = 0; gy < GridH; gy++)
                {
                    for (int gx = 0; gx < GridW; gx++)
                    {
                        float p1 = _pca.P1Min + (gx + 0.5f) / GridW * (_pca.P1Max - _pca.P1Min);
                        float p2 = _pca.P2Max - (gy + 0.5f) / GridH * (_pca.P2Max - _pca.P2Min);
                        _pca.Reconstruct(p1, p2, _probeBuffer);
                        ReadOnlySpan<float> probs = _probe.Run(network, _probeBuffer);
                        int cls = Metrics.ArgMax(probs);
                        g.FillRectangle(CellBrush(cls, probs[cls]),
                            plot.Left + gx * cellW, plot.Top + gy * cellH, cellW + 1, cellH + 1);
                    }
                }
            }

            for (int i = 0; i < _classes.Length; i++)
            {
                float x = plot.Left + (_projected[i * 2] - _pca.P1Min) / (_pca.P1Max - _pca.P1Min) * (plot.Width - 1);
                float y = plot.Top + (_pca.P2Max - _projected[i * 2 + 1]) / (_pca.P2Max - _pca.P2Min) * (plot.Height - 1);
                using var fill = new SolidBrush(ClassColors[_classes[i] % ClassColors.Length]);
                using var stroke = new Pen(UiTheme.TextPrimary, 1f);
                g.FillEllipse(fill, x - 3, y - 3, 6, 6);
                g.DrawEllipse(stroke, x - 3, y - 3, 6, 6);
            }

            using var frame = new Pen(UiTheme.Border);
            g.DrawRectangle(frame, plot);
            using var muted = new SolidBrush(UiTheme.TextMuted);
            g.DrawString("PC1", UiTheme.UiFont, muted, plot.Right - 24, plot.Bottom + 8);
            g.DrawString("PC2", UiTheme.UiFont, muted, 8, plot.Top - 2);
        }

        private SolidBrush CellBrush(int cls, float confidence)
        {
            _cellBrushes ??= new SolidBrush?[ClassColors.Length, 17];
            int level = Math.Clamp((int)(confidence * 16), 0, 16);
            int colorIndex = cls % ClassColors.Length;
            SolidBrush? brush = _cellBrushes[colorIndex, level];
            if (brush is null)
            {
                Color color = ClassColors[colorIndex];
                brush = new SolidBrush(Color.FromArgb(26 + level * 6, color));
                _cellBrushes[colorIndex, level] = brush;
            }

            return brush;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing && _cellBrushes is not null)
            {
                foreach (SolidBrush? brush in _cellBrushes)
                {
                    brush?.Dispose();
                }

                _cellBrushes = null;
            }

            base.Dispose(disposing);
        }
    }
}
