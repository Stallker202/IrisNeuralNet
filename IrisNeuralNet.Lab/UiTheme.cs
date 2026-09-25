using System;
using System.Drawing;

namespace IrisNeuralNet.Lab;

/// <summary>Единая визуальная тема лаборатории: все цвета и шрифты в одном месте.</summary>
internal static class UiTheme
{
    public static readonly Color Background = Color.FromArgb(24, 26, 31);
    public static readonly Color PanelBackground = Color.FromArgb(32, 35, 41);
    public static readonly Color ChartBackground = Color.FromArgb(18, 19, 23);
    public static readonly Color GridLine = Color.FromArgb(48, 52, 60);
    public static readonly Color Border = Color.FromArgb(58, 63, 73);
    public static readonly Color TextPrimary = Color.FromArgb(232, 234, 238);
    public static readonly Color TextMuted = Color.FromArgb(150, 156, 168);
    public static readonly Color AccentGreen = Color.FromArgb(94, 201, 121);
    public static readonly Color AccentBlue = Color.FromArgb(77, 171, 247);
    public static readonly Color AccentOrange = Color.FromArgb(240, 150, 66);
    public static readonly Color AccentGold = Color.FromArgb(232, 198, 87);
    public static readonly Color AccentRed = Color.FromArgb(224, 108, 117);
    public static readonly Color PreviousRun = Color.FromArgb(110, 116, 130);
    public static readonly Color PreviousRunDim = Color.FromArgb(86, 91, 103);

    public static readonly Font UiFont = new Font("Segoe UI", 9f);
    public static readonly Font MonoFont = new Font("Consolas", 9f);
    public static readonly Font TitleFont = new Font("Segoe UI", 9.5f, FontStyle.Bold);

    public static Color Mix(Color a, Color b, float t)
    {
        t = Math.Clamp(t, 0f, 1f);
        return Color.FromArgb(
            (int)(a.R + (b.R - a.R) * t),
            (int)(a.G + (b.G - a.G) * t),
            (int)(a.B + (b.B - a.B) * t));
    }
}