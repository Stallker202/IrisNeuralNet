using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace IrisNeuralNet.Lab.Reporting;

/// <summary>Таблица результатов: выравненный текст для консоли и Markdown для results.md.</summary>
internal sealed class TextTable
{
    private readonly string[] _headers;
    private readonly List<string[]> _rows = new();

    public TextTable(params string[] headers)
    {
        if (headers is null || headers.Length == 0) throw new ArgumentException("At least one column is required.", nameof(headers));
        _headers = headers;
    }

    public int RowCount => _rows.Count;

    public TextTable Add(params string[] cells)
    {
        if (cells.Length != _headers.Length)
        {
            throw new ArgumentException($"Expected {_headers.Length} cells, got {cells.Length}.", nameof(cells));
        }

        _rows.Add(cells);
        return this;
    }

    public string ToConsole()
    {
        int[] widths = _headers.Select((h, i) => Math.Max(h.Length, _rows.Count == 0 ? 0 : _rows.Max(r => r[i].Length))).ToArray();
        var text = new StringBuilder();
        AppendLine(text, _headers, widths);
        text.AppendLine(string.Join("-+-", widths.Select(w => new string('-', w))));
        foreach (string[] row in _rows) AppendLine(text, row, widths);
        return text.ToString().TrimEnd();
    }

    public string ToMarkdown()
    {
        var text = new StringBuilder();
        text.Append("| ").Append(string.Join(" | ", _headers.Select(Escape))).AppendLine(" |");
        text.Append('|').Append(string.Concat(_headers.Select(_ => "---|"))).AppendLine();
        foreach (string[] row in _rows)
        {
            text.Append("| ").Append(string.Join(" | ", row.Select(Escape))).AppendLine(" |");
        }

        return text.ToString().TrimEnd();
    }

    private static void AppendLine(StringBuilder text, IReadOnlyList<string> cells, int[] widths) =>
        text.AppendLine(string.Join(" | ", cells.Select((c, i) => c.PadRight(widths[i]))).TrimEnd());

    private static string Escape(string cell) => cell.Replace("|", "\\|");
}
