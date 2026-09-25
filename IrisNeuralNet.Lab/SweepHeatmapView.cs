using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace IrisNeuralNet.Lab
{
    internal sealed class SweepHeatmapView : PictureBox
    {
        private readonly Dictionary<long, SweepResult> _results = new();
        private int[] _neuronOptions = Array.Empty<int>();
        private int[] _layerOptions = Array.Empty<int>();
        private SweepResult _best;
        private bool _hasBest;

        public SweepHeatmapView()
        {
            SetStyle(ControlStyles.OptimizedDoubleBuffer | ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint, true);
            UpdateStyles();
        }

        public void UpdateData(IReadOnlyList<SweepResult> results, int[] neuronOptions, int[] layerOptions)
        {
            _results.Clear();
            _neuronOptions = neuronOptions;
            _layerOptions = layerOptions;
            _hasBest = false;
            foreach (SweepResult r in results)
            {
                _results[Key(r.HiddenLayers, r.Neurons)] = r;
                if (!_hasBest || r.ValidationAccuracy > _best.ValidationAccuracy)
                {
                    _best = r;
                    _hasBest = true;
                }
            }
        }

        private static long Key(int layers, int neurons) => layers * 100_000L + neurons;

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            Graphics g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            using (var bg = new SolidBrush(UiTheme.ChartBackground))
            {
                g.FillRectangle(bg, ClientRectangle);
            }

            using var titleBrush = new SolidBrush(UiTheme.TextPrimary);
            g.DrawString("Sweep: val accuracy по архитектурам (колонки — нейроны, строки — скрытые слои)",
                UiTheme.TitleFont, titleBrush, 8, 6);

            if (_neuronOptions.Length == 0 || _layerOptions.Length == 0)
            {
                using var waiting = new SolidBrush(UiTheme.TextMuted);
                g.DrawString("Нажми «Запустить sweep»...", UiTheme.UiFont, waiting, 10, 40);
                return;
            }

            Rectangle plot = Rectangle.FromLTRB(96, 34, ClientRectangle.Right - 16, ClientRectangle.Bottom - 16);
            if (plot.Width < 40 || plot.Height < 40)
            {
                return;
            }

            float cellW = plot.Width / (float)_neuronOptions.Length;
            float cellH = plot.Height / (float)_layerOptions.Length;

            using var headerBrush = new SolidBrush(UiTheme.TextMuted);
            for (int c = 0; c < _neuronOptions.Length; c++)
            {
                string text = _neuronOptions[c].ToString();
                SizeF size = g.MeasureString(text, UiTheme.UiFont);
                g.DrawString(text, UiTheme.UiFont, headerBrush, plot.Left + c * cellW + (cellW - size.Width) / 2, plot.Top - 18);
            }

            for (int r = 0; r < _layerOptions.Length; r++)
            {
                string text = $"слоёв: {_layerOptions[r]}";
                SizeF size = g.MeasureString(text, UiTheme.UiFont);
                g.DrawString(text, UiTheme.UiFont, headerBrush, plot.Left - size.Width - 8, plot.Top + r * cellH + (cellH - size.Height) / 2);
            }

            for (int r = 0; r < _layerOptions.Length; r++)
            {
                for (int c = 0; c < _neuronOptions.Length; c++)
                {
                    float x = plot.Left + c * cellW;
                    float y = plot.Top + r * cellH;
                    float w = cellW - 2;
                    float h = cellH - 2;

                    using (var empty = new SolidBrush(Color.FromArgb(26, 28, 33)))
                    {
                        g.FillRectangle(empty, x, y, w, h);
                    }

                    if (!_results.TryGetValue(Key(_layerOptions[r], _neuronOptions[c]), out SweepResult result))
                    {
                        continue;
                    }

                    using (var fill = new SolidBrush(Color.FromArgb(210, AccuracyColor(result.ValidationAccuracy))))
                    {
                        g.FillRectangle(fill, x, y, w, h);
                    }

                    string text = result.ValidationAccuracy.ToString("P0");
                    SizeF size = g.MeasureString(text, UiTheme.MonoFont);
                    using (var textBrush = new SolidBrush(UiTheme.Background))
                    {
                        g.DrawString(text, UiTheme.MonoFont, textBrush, x + (w - size.Width) / 2, y + (h - size.Height) / 2);
                    }

                    if (_hasBest && result.HiddenLayers == _best.HiddenLayers && result.Neurons == _best.Neurons)
                    {
                        using var frame = new Pen(UiTheme.TextPrimary, 2f);
                        g.DrawRectangle(frame, x + 1, y + 1, w - 2, h - 2);
                    }
                }
            }
        }

        private static Color AccuracyColor(float accuracy)
        {
            float t = Math.Clamp((accuracy - 0.34f) / (1f - 0.34f), 0f, 1f);
            return t < 0.5f
                ? UiTheme.Mix(UiTheme.AccentRed, UiTheme.AccentGold, t * 2f)
                : UiTheme.Mix(UiTheme.AccentGold, UiTheme.AccentGreen, (t - 0.5f) * 2f);
        }
    }
}
