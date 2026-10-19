using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.Text;

namespace IrisNeuralNet.Lab.Rendering;

/// <summary>Роль текста: определяет шрифт и размер (как Font-объекты в прежнем GDI+-коде).</summary>
internal enum TextRole
{
    /// <summary>Обычный текст, 12 px.</summary>
    Label,

    /// <summary>Заголовок, 12.7 px полужирный.</summary>
    Title,

    /// <summary>Моноширинный, 12 px: числа на осях и в легенде.</summary>
    Mono,
}

internal enum TextAnchor
{
    Start,
    Middle,
    End,
}

/// <summary>
/// Минимальный построитель SVG. Координаты — пиксели, как в прежнем коде на System.Drawing;
/// текст позиционируется по верхнему краю (как Graphics.DrawString). Один и тот же код
/// рисует виды в браузере и графики в файлах отчётов, на любой ОС и без зависимостей.
/// </summary>
internal sealed class SvgBuilder
{
    private const float LabelSize = 12f;
    private const float TitleSize = 12.7f;

    /// <summary>Расстояние от верха строки до базовой линии в долях кегля.</summary>
    private const float BaselineRatio = 0.95f;

    private readonly StringBuilder _body = new();
    private readonly List<KeyValuePair<string, string>> _attributes = new();

    public SvgBuilder(float width, float height)
    {
        if (!(width > 0f) || !(height > 0f)) throw new ArgumentOutOfRangeException(nameof(width), "SVG size must be positive.");

        Width = width;
        Height = height;
    }

    public float Width { get; }
    public float Height { get; }

    /// <summary>Приблизительная ширина текста: точное измерение без шрифтового движка невозможно, для раскладки хватает оценки.</summary>
    public static float EstimateWidth(string text, TextRole role = TextRole.Label) =>
        text.Length * (role == TextRole.Mono ? 7.2f : role == TextRole.Title ? 7.4f : 6.6f);

    /// <summary>data-атрибут корневого элемента (например, данные для подсказки); повторный вызов заменяет значение.</summary>
    public SvgBuilder Data(string name, string value)
    {
        string key = "data-" + name;
        _attributes.RemoveAll(attribute => attribute.Key == key);
        _attributes.Add(new KeyValuePair<string, string>(key, value));
        return this;
    }

    public SvgBuilder Rect(RectangleF rect, Color? fill, Color? stroke = null, float strokeWidth = 1f) =>
        Rect(rect.X, rect.Y, rect.Width, rect.Height, fill, stroke, strokeWidth);

    public SvgBuilder Rect(float x, float y, float width, float height, Color? fill, Color? stroke = null, float strokeWidth = 1f)
    {
        if (!(width > 0f) || !(height > 0f)) return this;

        _body.Append("<rect x=\"").Append(F(x)).Append("\" y=\"").Append(F(y))
            .Append("\" width=\"").Append(F(width)).Append("\" height=\"").Append(F(height)).Append('"');
        AppendFill(fill);
        AppendStroke(stroke, strokeWidth);
        _body.Append("/>");
        return this;
    }

    public SvgBuilder Line(float x1, float y1, float x2, float y2, Color stroke, float width = 1f, bool dotted = false)
    {
        _body.Append("<line x1=\"").Append(F(x1)).Append("\" y1=\"").Append(F(y1))
            .Append("\" x2=\"").Append(F(x2)).Append("\" y2=\"").Append(F(y2)).Append('"');
        AppendStroke(stroke, width);
        if (dotted) _body.Append(" stroke-dasharray=\"1 2\"");
        _body.Append("/>");
        return this;
    }

    public SvgBuilder Polyline(IReadOnlyList<PointF> points, Color stroke, float width)
    {
        if (points.Count < 2) return this;

        _body.Append("<polyline fill=\"none\" stroke-linejoin=\"round\" stroke-linecap=\"round\" points=\"");
        for (int i = 0; i < points.Count; i++)
        {
            if (i > 0) _body.Append(' ');
            _body.Append(F(points[i].X)).Append(',').Append(F(points[i].Y));
        }

        _body.Append('"');
        AppendStroke(stroke, width);
        _body.Append("/>");
        return this;
    }

    public SvgBuilder Circle(float cx, float cy, float radius, Color? fill, Color? stroke = null, float strokeWidth = 1f)
    {
        _body.Append("<circle cx=\"").Append(F(cx)).Append("\" cy=\"").Append(F(cy)).Append("\" r=\"").Append(F(radius)).Append('"');
        AppendFill(fill);
        AppendStroke(stroke, strokeWidth);
        _body.Append("/>");
        return this;
    }

    /// <param name="top">Верх строки текста (как y в Graphics.DrawString).</param>
    public SvgBuilder Text(float x, float top, string text, Color color, TextRole role = TextRole.Label, TextAnchor anchor = TextAnchor.Start)
    {
        float size = role == TextRole.Title ? TitleSize : LabelSize;
        _body.Append("<text x=\"").Append(F(x)).Append("\" y=\"").Append(F(top + size * BaselineRatio)).Append('"');
        if (role == TextRole.Title) _body.Append(" class=\"t\"");
        else if (role == TextRole.Mono) _body.Append(" class=\"m\"");
        if (anchor != TextAnchor.Start) _body.Append(" text-anchor=\"").Append(anchor == TextAnchor.Middle ? "middle" : "end").Append('"');
        AppendFill(color);
        _body.Append('>');
        AppendEscaped(_body, text);
        _body.Append("</text>");
        return this;
    }

    /// <param name="pixelated">Масштабировать без сглаживания (изображения 28×28).</param>
    public SvgBuilder Image(float x, float y, float width, float height, string href, bool pixelated)
    {
        _body.Append("<image x=\"").Append(F(x)).Append("\" y=\"").Append(F(y))
            .Append("\" width=\"").Append(F(width)).Append("\" height=\"").Append(F(height)).Append("\" href=\"");
        AppendEscaped(_body, href);
        _body.Append('"');
        if (pixelated) _body.Append(" style=\"image-rendering:pixelated\"");
        _body.Append("/>");
        return this;
    }

    public override string ToString()
    {
        var svg = new StringBuilder(_body.Length + 512);
        svg.Append("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"").Append(F(Width)).Append("\" height=\"").Append(F(Height))
            .Append("\" viewBox=\"0 0 ").Append(F(Width)).Append(' ').Append(F(Height)).Append('"');
        foreach (KeyValuePair<string, string> attribute in _attributes)
        {
            svg.Append(' ').Append(attribute.Key).Append("=\"");
            AppendEscaped(svg, attribute.Value);
            svg.Append('"');
        }

        svg.Append("><style>text{font-family:").Append(Theme.UiFontFamily).Append(";font-size:").Append(F(LabelSize))
            .Append("px}.t{font-weight:700;font-size:").Append(F(TitleSize))
            .Append("px}.m{font-family:").Append(Theme.MonoFontFamily).Append("}</style>")
            .Append(_body)
            .Append("</svg>");
        return svg.ToString();
    }

    /// <summary>Число с одним знаком после точки, без культурных сюрпризов: 12.5, 7, -3.2.</summary>
    public static string F(float value) =>
        float.IsFinite(value) ? MathF.Round(value, 1).ToString("0.#", CultureInfo.InvariantCulture) : "0";

    private void AppendFill(Color? color)
    {
        if (color is not { } c)
        {
            _body.Append(" fill=\"none\"");
            return;
        }

        _body.Append(" fill=\"").Append(Theme.Hex(c)).Append('"');
        if (c.A < 255) _body.Append(" fill-opacity=\"").Append(Opacity(c)).Append('"');
    }

    private void AppendStroke(Color? color, float width)
    {
        if (color is not { } c) return;

        _body.Append(" stroke=\"").Append(Theme.Hex(c)).Append('"');
        if (c.A < 255) _body.Append(" stroke-opacity=\"").Append(Opacity(c)).Append('"');
        if (width != 1f) _body.Append(" stroke-width=\"").Append(F(width)).Append('"');
    }

    private static string Opacity(Color color) => (color.A / 255f).ToString("0.###", CultureInfo.InvariantCulture);

    private static void AppendEscaped(StringBuilder target, string text)
    {
        foreach (char c in text)
        {
            switch (c)
            {
                case '&': target.Append("&amp;"); break;
                case '<': target.Append("&lt;"); break;
                case '>': target.Append("&gt;"); break;
                case '"': target.Append("&quot;"); break;
                default: target.Append(c); break;
            }
        }
    }
}
