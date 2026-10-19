using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using IrisNeuralNet.Core;
using IrisNeuralNet.Diagnostics;
using IrisNeuralNet.Lab.Experiments;
using IrisNeuralNet.Lab.Reporting;
using IrisNeuralNet.Optimizers;

namespace IrisNeuralNet.Lab.Scenarios;

/// <summary>Сравнение моделей, выбор лучшей и словесный разбор переобучения — общие для всех лаб.</summary>
internal static class ExperimentAnalysis
{
    public static TextTable Comparison(IReadOnlyList<ExperimentResult> results)
    {
        string metric = results.Count > 0 ? results[0].Metric.Name : "metric";
        var table = new TextTable(
            "Модель", "Архитектура", "Оптимизатор", "Эпох", "Batch", "Параметров",
            $"train {metric}", $"val {metric}", $"лучшая val {metric} (эпоха)", "мин. val_loss (эпоха)", "Время");

        foreach (ExperimentResult r in results)
        {
            EpochMetrics best = r.BestValidation;
            EpochMetrics minLoss = r.MinValidationLoss;
            table.Add(
                r.Config.Name,
                r.Architecture,
                OptimizerFactory.KerasName(r.Config.Optimizer),
                r.History.Count.ToString(CultureInfo.InvariantCulture),
                r.Config.BatchSize.ToString(CultureInfo.InvariantCulture),
                r.Experiment.Network.ParameterCount.ToString("N0", CultureInfo.InvariantCulture),
                MetricFormat.Value(r.Metric, r.Train.Metric),
                MetricFormat.Value(r.Metric, r.Validation.Metric),
                $"{MetricFormat.Value(r.Metric, best.ValidationMetric)} ({best.Epoch})",
                $"{MetricFormat.Loss(minLoss.ValidationLoss)} ({minLoss.Epoch})",
                ExperimentRunner.Seconds(r.Elapsed));
        }

        return table;
    }

    /// <summary>
    /// Лучшая модель — по финальной валидационной метрике (как советует методичка: она показывает
    /// обобщение на новые данные); при равенстве — меньший val_loss (модель увереннее в верных ответах),
    /// затем меньше параметров (бритва Оккама).
    /// </summary>
    public static ExperimentResult SelectBest(IReadOnlyList<ExperimentResult> results)
    {
        if (results.Count == 0) throw new ArgumentException("No results to compare.", nameof(results));

        Metric metric = results[0].Metric;
        return results
            .OrderByDescending(r => metric.HigherIsBetter ? r.Validation.Metric : -r.Validation.Metric)
            .ThenBy(r => r.Validation.Loss)
            .ThenBy(r => r.Experiment.Network.ParameterCount)
            .First();
    }

    /// <summary>Разрыв между train и validation в сторону «train лучше».</summary>
    public static float Gap(ExperimentResult result) =>
        result.Metric.HigherIsBetter
            ? result.Train.Metric - result.Validation.Metric
            : result.Validation.Metric - result.Train.Metric;

    /// <summary>
    /// Словесный диагноз по кривым: не обучилась / переобучение / слабые признаки / норма.
    /// Конец кривой усредняется по последним 10% эпох, чтобы шум маленькой валидации
    /// (30 образцов у Iris) не выдавался за переобучение.
    /// </summary>
    public static string Overfitting(ExperimentResult result)
    {
        IReadOnlyList<EpochMetrics> epochs = result.History.Epochs;
        EpochMetrics first = epochs[0];
        EpochMetrics last = epochs[^1];
        string gap = result.Metric.HigherIsBetter
            ? $"разрыв accuracy train−val {Gap(result) * 100f:F1} п.п."
            : $"разрыв MAE val−train {Gap(result):F3}";

        int classCount = result.Experiment.Data.ClassCount;
        if (classCount > 0 && result.Train.Metric < 1f / classCount + 0.1f)
        {
            return $"{result.Config.Name}: модель не обучилась — точность на обучении {MetricFormat.Value(result.Metric, result.Train.Metric)} " +
                   $"на уровне случайного угадывания (1/{classCount}); потери {MetricFormat.Loss(first.TrainLoss)} → {MetricFormat.Loss(last.TrainLoss)}. " +
                   "Типичная причина — «мёртвые» ReLU: при положительном ненормализованном входе скрытые нейроны уходят в z < 0 и перестают " +
                   $"получать градиент; лечится нормализацией входа или другим seed/learning rate; {gap}";
        }

        int tail = Math.Max(3, epochs.Count / 10);
        EpochMetrics minLoss = result.MinValidationLoss;
        float endLoss = epochs.Skip(Math.Max(0, epochs.Count - tail)).Average(m => m.ValidationLoss);
        float rise = (endLoss - minLoss.ValidationLoss) / Math.Max(minLoss.ValidationLoss, 1e-6f);
        bool minimumWellBeforeEnd = minLoss.Epoch <= last.Epoch - tail;

        string verdict = (rise, minimumWellBeforeEnd) switch
        {
            ( > 0.15f, true) => $"переобучение: val_loss минимален на эпохе {minLoss.Epoch} ({MetricFormat.Loss(minLoss.ValidationLoss)}), " +
                                $"в последних {tail} эпохах в среднем выше на {rise:P0}",
            ( > 0.05f, true) => $"слабые признаки переобучения: после эпохи {minLoss.Epoch} val_loss вырос в среднем на {rise:P0}",
            _ => "явных признаков переобучения нет: val_loss к концу обучения у своего минимума",
        };

        return $"{result.Config.Name}: {verdict}; {gap}";
    }

    /// <summary>Сравнение модели с эталонной: изменение валидационной метрики и времени обучения.</summary>
    public static string Versus(ExperimentResult reference, ExperimentResult candidate)
    {
        Metric metric = reference.Metric;
        float delta = candidate.Validation.Metric - reference.Validation.Metric;
        string change = metric.HigherIsBetter
            ? (delta * 100f).ToString("+0.0;-0.0;0.0", CultureInfo.InvariantCulture) + " п.п."
            : delta.ToString("+0.000;-0.000;0.000", CultureInfo.InvariantCulture);
        string verdict = Math.Abs(delta) < 1e-6f ? "без изменений"
            : metric.IsBetter(candidate.Validation.Metric, reference.Validation.Metric) ? "лучше" : "хуже";

        return $"«{candidate.Config.Name}»: val {metric.Name} {MetricFormat.Value(metric, candidate.Validation.Metric)} " +
               $"против {MetricFormat.Value(metric, reference.Validation.Metric)} у «{reference.Config.Name}» ({change}, {verdict}); " +
               $"время {ExperimentRunner.Seconds(candidate.Elapsed)} против {ExperimentRunner.Seconds(reference.Elapsed)}.";
    }

    /// <summary>Ряд валидационной метрики каждой модели — для общего графика сравнения.</summary>
    public static PlotSeries[] ValidationSeries(IEnumerable<ExperimentResult> results) =>
        results.Select(r => new PlotSeries(r.Config.Name, r.History.Series(m => m.ValidationMetric))).ToArray();
}
