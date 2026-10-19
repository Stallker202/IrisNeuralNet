using System.Collections.Generic;
using IrisNeuralNet.Diagnostics;

namespace IrisNeuralNet.Lab.Reporting;

/// <summary>
/// Сохранение графиков для отчёта (аналог matplotlib в методичках). Сценарии зависят только
/// от этого интерфейса: SVG рисует Rendering.SvgChartRenderer на любой ОС, --no-charts отключает графики.
/// </summary>
internal interface IChartRenderer
{
    /// <summary>Расширение файлов (".svg") или null, если рендерер ничего не сохраняет.</summary>
    string? Extension { get; }

    /// <summary>Два графика рядом, как plot_training_history: потери и метрика, train и validation.</summary>
    void SaveHistory(string path, string title, TrainingHistory history, string metricLabel);

    /// <summary>Несколько рядов на одном графике (сравнение моделей, усреднённая MAE K-Fold).</summary>
    void SaveLines(string path, string title, string yLabel, IReadOnlyList<PlotSeries> series);

    /// <summary>Сетка изображений цифр с подписями (аналог plt.imshow).</summary>
    void SaveDigits(string path, string title, IReadOnlyList<DigitSample> samples);
}
