using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using IrisNeuralNet.Data;
using IrisNeuralNet.Diagnostics;
using IrisNeuralNet.Lab.Experiments;
using IrisNeuralNet.Lab.Reporting;
using IrisNeuralNet.Optimizers;

namespace IrisNeuralNet.Lab.Scenarios;

/// <summary>Лабораторная «Регрессия — предсказание цен на жильё в Бостоне (Boston Housing)».</summary>
internal sealed class BostonScenario : ILabScenario
{
    private const int Folds = 4;
    private const float ThousandDollars = 1000f;

    public string Key => "boston";

    public string Title => "Лабораторная: регрессия — цены на жильё в Бостоне (Boston Housing)";

    public void Run(ScenarioContext context)
    {
        DatasetSpec spec = Datasets.BostonHousing;

        context.Section("3. Загрузка и анализ данных");
        DatasetSource source = spec.Load();
        PreparedData raw = DataPreparation.Prepare(spec, source, FeatureScaling.None, validationFraction: 0.2f);
        Dataset train = raw.RawTrain;
        Dataset test = raw.RawValidation;
        context.Text($"Форма обучающих данных: ({train.RowCount}, {train.FeatureCount}), тестовых: ({test.RowCount}, {test.FeatureCount}); " +
                     $"меток: ({train.RowCount},) и ({test.RowCount},). Разбиение 80/20 как в keras.datasets.boston_housing (seed {DataPreparation.DefaultSplitSeed}).");
        float[] prices = train.Labels.ToArray();
        context.Text($"Первые 5 цен в обучающем наборе: [{string.Join(", ", prices.Take(5).Select(p => p.ToString("0.0", CultureInfo.InvariantCulture)))}] (тыс. $)");
        context.Text($"Диапазон цен: {prices.Min().ToString("0.0", CultureInfo.InvariantCulture)} – {prices.Max().ToString("0.0", CultureInfo.InvariantCulture)} тыс. $");
        context.Table(FeatureStatistics("До нормализации", train.Features.ToArray(), train.FeatureCount, train.RowCount));

        context.Section("4. Предобработка: нормализация (z-score по обучающей выборке)");
        PreparedData data = DataPreparation.Prepare(spec, source, FeatureScaling.Standardize, validationFraction: 0.2f);
        context.Text("μ и σ считаются только по обучающей выборке и применяются к обеим — статистика теста заранее неизвестна.");
        context.Table(FeatureStatistics("После нормализации (≈0 и ≈1)", data.TrainFeatures, data.FeatureCount, data.TrainRowCount));

        context.Section("5–6. Базовая модель: 13 → 64 → 64 → 1 (relu, linear), rmsprop, mse, метрика mae");
        int baseEpochs = context.Epochs(100);
        int shortEpochs = context.Epochs(50);
        var baseline = new ExperimentConfig($"Базовая модель ({RussianPlural.Epochs(baseEpochs)})", new[] { 64, 64 }, OptimizerKind.RmsProp, 0.001f, baseEpochs, BatchSize: 16);
        ExperimentResult base100 = ExperimentRunner.Run(context, data, baseline);
        ReportTestError(context, base100);

        context.Section($"8. Эксперимент: K-Fold перекрёстная проверка (K = {Folds})");
        // Как в листинге: блоки режутся из уже нормализованной обучающей выборки.
        var normalizedTrain = new Dataset(data.TrainFeatures, train.Labels.ToArray(), train.FeatureCount, classCount: 0);
        CrossValidationResult cv = CrossValidation.Run(context, spec, normalizedTrain, baseline with { Name = "K-Fold" }, Folds, FeatureScaling.None);
        float[] averageMae = cv.AverageValidationMetric();
        float[] finalMae = cv.FinalValidationMetrics();
        int tail = Math.Min(10, averageMae.Length);
        float lastTenMae = averageMae.Skip(averageMae.Length - tail).Average();
        float foldSpread = CrossValidation.StandardDeviation(finalMae);
        context.ChartLines("kfold-average-mae", $"Усреднённая MAE по {Folds} блокам (K-Fold)", "MAE",
            new[] { new PlotSeries("Средняя MAE (K-Fold CV)", averageMae) });
        context.Text(
            $"Средняя MAE за последние {tail} эпох (K-Fold): {lastTenMae.ToString("F4", CultureInfo.InvariantCulture)} " +
            $"({Dollars(lastTenMae)}); MAE блоков: " +
            $"[{string.Join(", ", finalMae.Select(m => m.ToString("F2", CultureInfo.InvariantCulture)))}], σ = {foldSpread.ToString("F2", CultureInfo.InvariantCulture)}.");

        context.Section($"9. Эксперимент: изменение количества эпох ({shortEpochs})");
        ExperimentResult base50 = ExperimentRunner.Run(context, data, baseline with { Name = $"Модель ({RussianPlural.Epochs(shortEpochs)})", Epochs = shortEpochs }, showSummary: false);
        ReportTestError(context, base50);

        context.Section("10. Сравнение и анализ результатов");
        var results = new List<ExperimentResult> { base100, base50 };
        context.Table(ExperimentAnalysis.Comparison(results));
        var summary = new TextTable("Оценка", "MAE", "MAE, $")
            .Add($"Базовая модель, тест ({RussianPlural.Epochs(baseEpochs)})", F(base100.Validation.Metric), Dollars(base100.Validation.Metric))
            .Add($"Модель {RussianPlural.Epochs(shortEpochs)}, тест", F(base50.Validation.Metric), Dollars(base50.Validation.Metric))
            .Add($"K-Fold, среднее последних {tail} эпох", F(lastTenMae), Dollars(lastTenMae));
        context.Table(summary);

        EpochMetrics bestEpoch = base100.BestValidation;
        context.Bullet(
            $"Где начинается переобучение: у базовой модели val_mae минимальна на эпохе {bestEpoch.Epoch} ({F(bestEpoch.ValidationMetric)}), " +
            $"val_loss — на эпохе {base100.MinValidationLoss.Epoch}; к последней эпохе val_mae {F(base100.History.Last.ValidationMetric)}, " +
            $"train_mae {F(base100.History.Last.TrainMetric)}. Диагноз по кривым — {ExperimentAnalysis.Overfitting(base100)}");
        context.Bullet(
            $"Стабильность K-Fold: MAE отдельных блоков различаются (σ = {foldSpread:F2} тыс. $) — ровно настолько «гуляла» бы оценка по одной случайной " +
            $"тестовой выборке из ~100 домов. Среднее по {Folds} блокам надёжнее одиночного evaluate.");
        context.Bullet(
            $"Влияние числа эпох: {RussianPlural.Epochs(shortEpochs)} → MAE {F(base50.Validation.Metric)}, {RussianPlural.Epochs(baseEpochs)} → {F(base100.Validation.Metric)}. " +
            ExperimentAnalysis.Versus(base100, base50));
        context.Text(
            "Замечание к методике: в листинге validation_data = тестовая выборка, поэтому выбор числа эпох по её графику — подглядывание в тест. " +
            "Честнее подбирать эпохи по K-Fold на обучающей выборке и только в конце один раз оценить модель на тесте.");
        ModelCheckpoint.SaveAndVerify(context, base100, "boston_model.nn");
    }

    private static void ReportTestError(ScenarioContext context, ExperimentResult result) =>
        context.Text(
            $"--- Оценка модели «{result.Config.Name}» на тестовых данных --- Среднеквадратичная ошибка (MSE): {F(result.Validation.Loss)}; " +
            $"средняя абсолютная ошибка (MAE): {F(result.Validation.Metric)} = {Dollars(result.Validation.Metric)}");

    private static TextTable FeatureStatistics(string caption, float[] features, int featureCount, int rowCount)
    {
        var table = new TextTable(caption, "Среднее", "Ст. отклонение");
        for (int f = 0; f < 5; f++)
        {
            double mean = 0;
            for (int r = 0; r < rowCount; r++) mean += features[r * featureCount + f];
            mean /= rowCount;

            double variance = 0;
            for (int r = 0; r < rowCount; r++)
            {
                double diff = features[r * featureCount + f] - mean;
                variance += diff * diff;
            }

            table.Add(
                BostonHousingDataLoader.FeatureNames[f],
                mean.ToString("F3", CultureInfo.InvariantCulture),
                Math.Sqrt(variance / rowCount).ToString("F3", CultureInfo.InvariantCulture));
        }

        return table;
    }

    private static string F(float value) => value.ToString("F2", CultureInfo.InvariantCulture);

    private static string Dollars(float maeThousands) => "$" + (maeThousands * ThousandDollars).ToString("N0", CultureInfo.InvariantCulture);
}
