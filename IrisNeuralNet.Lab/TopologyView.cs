using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;
using IrisNeuralNet.Core;

namespace IrisNeuralNet.Lab
{
    internal sealed class TopologyView : PictureBox
    {
        private readonly NetworkProbe _probe = new();
        private NeuralNetwork? _network;
        private float[] _probeSample = Array.Empty<float>();

        public TopologyView()
        {
            SetStyle(ControlStyles.OptimizedDoubleBuffer | ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint, true);
            UpdateStyles();
        }

        public void UpdateSource(NeuralNetwork? network, float[] probeSample)
        {
            _network = network;
            _probeSample = probeSample;
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            Graphics g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            using (var bg = new SolidBrush(UiTheme.ChartBackground))
            {
                g.FillRectangle(bg, ClientRectangle);
            }

            NeuralNetwork? network = _network;
            if (network is null || _probeSample.Length == 0)
            {
                using var waiting = new SolidBrush(UiTheme.TextMuted);
                g.DrawString("ожидание сеть...", UiTheme.UiFont, waiting, 10, 10);
                return;
            }

            _probe.Run(network, _probeSample);

            using var title = new SolidBrush(UiTheme.TextPrimary);
            g.DrawString("Топология: веса и активации (проба — образец 0)", UiTheme.TitleFont, title, 8, 6);

            int columns = network.LayerCount + 1;
            Rectangle area = Rectangle.FromLTRB(16, 30, ClientRectangle.Right - 16, ClientRectangle.Bottom - 34);
            if (area.Width < 40 || area.Height < 40)
            {
                return;
            }

            int[] counts = new int[columns];
            counts[0] = network.InputDim;
            for (int i = 0; i < network.LayerCount; i++)
            {
                counts[i + 1] = network.Layers[i].OutputDim;
            }

            float[] columnX = new float[columns];
            for (int c = 0; c < columns; c++)
            {
                columnX[c] = area.Left + c * (area.Width - 1) / (float)(columns - 1);
            }

            float radius = 9f;
            for (int c = 0; c < columns; c++)
            {
                radius = MathF.Min(radius, area.Height / (2.4f * counts[c]));
            }

            radius = MathF.Max(radius, 2.5f);

            for (int i = 0; i < network.LayerCount; i++)
            {
                DenseLayer layer = network.Layers[i];
                ReadOnlySpan<float> w = layer.Weights;
                float maxAbs = 1e-6f;
                for (int k = 0; k < w.Length; k++)
                {
                    maxAbs = MathF.Max(maxAbs, MathF.Abs(w[k]));
                }

                for (int inN = 0; inN < layer.InputDim; inN++)
                {
                    for (int outN = 0; outN < layer.OutputDim; outN++)
                    {
                        float weight = w[inN * layer.OutputDim + outN];
                        float t = MathF.Abs(weight) / maxAbs;
                        if (t < 0.04f)
                        {
                            continue;
                        }

                        Color color = weight >= 0f ? UiTheme.AccentBlue : UiTheme.AccentRed;
                        using var pen = new Pen(
                            Color.FromArgb((int)(30 + 150 * t), color),
                            0.6f + 2.2f * t);
                        g.DrawLine(pen,
                            Position(columnX, area, i, inN, counts[i]),
                            Position(columnX, area, i + 1, outN, counts[i + 1]));
                    }
                }
            }

            for (int c = 0; c < columns; c++)
            {
                for (int n = 0; n < counts[c]; n++)
                {
                    float value = c == 0 ? _probeSample[n] : _probe.Activations[c - 1][n];
                    float squashed = value / (1f + MathF.Abs(value));
                    float t = (squashed + 1f) / 2f;
                    PointF p = Position(columnX, area, c, n, counts[c]);
                    using var fill = new SolidBrush(UiTheme.Mix(UiTheme.PanelBackground, UiTheme.AccentGreen, t));
                    using var stroke = new Pen(c == 0 ? UiTheme.TextMuted : UiTheme.Border);
                    g.FillEllipse(fill, p.X - radius, p.Y - radius, radius * 2, radius * 2);
                    g.DrawEllipse(stroke, p.X - radius, p.Y - radius, radius * 2, radius * 2);
                }
            }

            using var muted = new SolidBrush(UiTheme.TextMuted);
            for (int c = 0; c < columns; c++)
            {
                string label = c == 0
                    ? $"вход {counts[0]}"
                    : $"L{c} {network.Layers[c - 1].ActivationKind} {counts[c]}";
                SizeF size = g.MeasureString(label, UiTheme.UiFont);
                g.DrawString(label, UiTheme.UiFont, muted, columnX[c] - size.Width / 2, area.Bottom + 8);
            }
        }

        private static PointF Position(float[] columnX, Rectangle area, int column, int index, int count) =>
            new PointF(
                columnX[column],
                area.Top + (count == 1 ? area.Height / 2f : index * (area.Height - 1) / (float)(count - 1)));
    }
}
