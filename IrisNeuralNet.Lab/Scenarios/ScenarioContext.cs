using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using IrisNeuralNet.Diagnostics;
using IrisNeuralNet.Lab.Reporting;

namespace IrisNeuralNet.Lab.Scenarios;

/// <summary>Загрузка пользовательского изображения в оттенки серого (0 — чёрный, 255 — белый).</summary>
internal interface IImageLoader
{
    float[] LoadGrayscale(string path, out int width, out int height);
}

/// <summary>Параметры запуска сценариев из командной строки.</summary>
internal sealed record ScenarioOptions(bool Quick, IChartRenderer Charts, IImageLoader? Images, string? ImagePath);

/// <summary>
/// Вывод сценария лабораторной: каждая запись одновременно идёт в консоль и в results.md,
/// графики сохраняются рядом и вставляются в Markdown ссылкой. Один вызов — оба канала (DRY).
/// </summary>
internal sealed class ScenarioContext
{
    private readonly StringBuilder _markdown = new();
    private readonly string _title;
    private bool _inList;

    public ScenarioContext(string title, string reportDirectory, ScenarioOptions options, TextWriter output)
    {
        _title = title;
        ReportDirectory = reportDirectory;
        Options = options;
        Output = output;
        Directory.CreateDirectory(reportDirectory);
    }

    public ScenarioOptions Options { get; }
    public string ReportDirectory { get; }
    public TextWriter Output { get; }

    /// <summary>В режиме --quick (smoke-тест) эпох в 10 раз меньше.</summary>
    public int Epochs(int full) => Options.Quick ? Math.Max(1, (full + 9) / 10) : full;

    public string PathOf(string fileName) => Path.Combine(ReportDirectory, fileName);

    public void Section(string title)
    {
        EndList();
        Output.WriteLine();
        Output.WriteLine($"=== {title} ===");
        _markdown.AppendLine().AppendLine($"## {title}").AppendLine();
    }

    public void Text(string text)
    {
        EndList();
        Output.WriteLine(text);
        _markdown.AppendLine(text).AppendLine();
    }

    public void Bullet(string text)
    {
        Output.WriteLine($"  • {text}");
        _markdown.AppendLine($"- {text}");
        _inList = true;
    }

    public void Code(string text)
    {
        EndList();
        Output.WriteLine(text);
        _markdown.AppendLine("```").AppendLine(text).AppendLine("```").AppendLine();
    }

    public void Table(TextTable table)
    {
        EndList();
        Output.WriteLine(table.ToConsole());
        _markdown.AppendLine(table.ToMarkdown()).AppendLine();
    }

    public void ChartHistory(string fileStem, string title, TrainingHistory history, string metricLabel) =>
        Chart(fileStem, title, path => Options.Charts.SaveHistory(path, title, history, metricLabel));

    public void ChartLines(string fileStem, string title, string yLabel, IReadOnlyList<PlotSeries> series) =>
        Chart(fileStem, title, path => Options.Charts.SaveLines(path, title, yLabel, series));

    public void ChartDigits(string fileStem, string title, IReadOnlyList<DigitSample> samples) =>
        Chart(fileStem, title, path => Options.Charts.SaveDigits(path, title, samples));

    /// <summary>Записывает results.md и возвращает путь к нему.</summary>
    public string Save()
    {
        string path = PathOf("results.md");
        File.WriteAllText(path, $"# {_title}\n{_markdown}", new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        return path;
    }

    /// <summary>Пустая строка после списка: иначе следующий абзац Markdown «приклеится» к последнему пункту.</summary>
    private void EndList()
    {
        if (!_inList) return;
        _markdown.AppendLine();
        _inList = false;
    }

    private void Chart(string fileStem, string title, Action<string> render)
    {
        string? extension = Options.Charts.Extension;
        if (extension is null) return;

        EndList();
        string fileName = fileStem + extension;
        render(PathOf(fileName));
        Output.WriteLine($"  [график] {fileName}");
        _markdown.AppendLine($"![{title}]({fileName})").AppendLine();
    }
}
