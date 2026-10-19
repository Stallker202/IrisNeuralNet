using System;
using System.Drawing;
using System.Globalization;
using System.Text;

namespace IrisNeuralNet.Lab.Rendering;

/// <summary>
/// Единая визуальная тема лаборатории (та же палитра, что была в WinForms-версии).
/// Отсюда же генерируются CSS-переменные страницы, поэтому цвета заданы в одном месте.
/// </summary>
internal static class Theme
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
    public static readonly Color Hover = Color.FromArgb(48, 52, 60);
    public static readonly Color EmptyCell = Color.FromArgb(26, 28, 33);

    /// <summary>Цвета классов: границы решений и точки датасета.</summary>
    public static readonly Color[] ClassColors =
    {
        AccentBlue, AccentGreen, AccentOrange, AccentGold, AccentRed,
        Color.FromArgb(177, 132, 230), Color.FromArgb(94, 205, 205), Color.FromArgb(240, 128, 190),
        Color.FromArgb(160, 170, 90), Color.FromArgb(200, 160, 120),
    };

    /// <summary>Segoe UI на Windows, системный шрифт на macOS и Linux.</summary>
    public const string UiFontFamily = "'Segoe UI', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif";

    public const string MonoFontFamily = "Consolas, Menlo, 'DejaVu Sans Mono', monospace";

    public static Color Mix(Color a, Color b, float t)
    {
        t = float.IsNaN(t) ? 0f : Math.Clamp(t, 0f, 1f);
        return Color.FromArgb(
            (int)(a.R + (b.R - a.R) * t),
            (int)(a.G + (b.G - a.G) * t),
            (int)(a.B + (b.B - a.B) * t));
    }

    public static string Hex(Color color) => $"#{color.R:x2}{color.G:x2}{color.B:x2}";

    /// <summary>CSS-переменные для страницы: --background, --accent-green, … (имя поля в kebab-case).</summary>
    public static string ToCss()
    {
        var css = new StringBuilder(":root {\n");
        AppendVariable(css, "background", Background);
        AppendVariable(css, "panel-background", PanelBackground);
        AppendVariable(css, "chart-background", ChartBackground);
        AppendVariable(css, "grid-line", GridLine);
        AppendVariable(css, "border", Border);
        AppendVariable(css, "text-primary", TextPrimary);
        AppendVariable(css, "text-muted", TextMuted);
        AppendVariable(css, "accent-green", AccentGreen);
        AppendVariable(css, "accent-blue", AccentBlue);
        AppendVariable(css, "accent-orange", AccentOrange);
        AppendVariable(css, "accent-gold", AccentGold);
        AppendVariable(css, "accent-red", AccentRed);
        AppendVariable(css, "hover", Hover);
        css.Append(CultureInfo.InvariantCulture, $"  --font-ui: {UiFontFamily};\n");
        css.Append(CultureInfo.InvariantCulture, $"  --font-mono: {MonoFontFamily};\n");
        return css.Append("}\n").ToString();
    }

    private static void AppendVariable(StringBuilder css, string name, Color color) =>
        css.Append(CultureInfo.InvariantCulture, $"  --{name}: {Hex(color)};\n");
}

/// <summary>Палитра графика: тёмная — для страницы лаборатории, светлая — для отчётов.</summary>
internal sealed record ChartStyle(Color Background, Color Text, Color MutedText, Color Grid, Color Border)
{
    public static readonly ChartStyle Dark = new(Theme.ChartBackground, Theme.TextPrimary, Theme.TextMuted, Theme.GridLine, Theme.Border);

    public static readonly ChartStyle Light = new(
        Color.White, Color.FromArgb(33, 37, 41), Color.FromArgb(96, 103, 112), Color.FromArgb(226, 229, 233), Color.FromArgb(173, 181, 189));
}
