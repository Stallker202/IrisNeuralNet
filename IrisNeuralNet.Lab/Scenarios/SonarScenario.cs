using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using IrisNeuralNet.Data;
using IrisNeuralNet.Lab.Experiments;
using IrisNeuralNet.Lab.Reporting;
using IrisNeuralNet.Optimizers;

namespace IrisNeuralNet.Lab.Scenarios;

/// <summary>Лабораторная «Бинарная классификация сигналов радара (Sonar)».</summary>
internal sealed class SonarScenario : ILabScenario
{
    private const int CrossValidationFolds = 5;

    public string Key => "sonar";

    public string Title => "Лабораторная: бинарная классификация сигналов радара (Sonar)";

    public void Run(ScenarioContext context)
    {
        DatasetSpec spec = Datasets.Sonar;

        context.Section("3. Загрузка и анализ данных");
        DatasetSource source = spec.Load();
        Dataset all = source.Train;
        int metals = all.ClassIndices().Count(c => c == 1);
        context.Text($"dataframe.shape: ({all.RowCount}, {all.FeatureCount + 1}) → X: ({all.RowCount}, {all.FeatureCount}), Y_labels: ({all.RowCount},)");
        context.Text($"Уникальные метки: ['M' 'R']; камней (R): {all.RowCount - metals}, металла (M): {metals} — классы почти сбалансированы.");
        context.Code("head(): первые 5 признаков\n" + string.Join("\n", Enumerable.Range(0, 5).Select(i => FormatHead(all, i))));

        context.Section("4. Предобработка данных");
        context.Text(
            "Кодирование меток: R → 0, M → 1 (как в методичке; sklearn.LabelEncoder отсортировал бы ['M', 'R'] и дал M → 0 — на качество не влияет). " +
            "One-hot не нужен: sigmoid + binary_crossentropy работают с меткой 0/1 напрямую.");
        context.Text($"Закодированные метки (первые 10): [{string.Join(" ", all.Labels.Slice(0, 10).ToArray().Select(v => v.ToString(CultureInfo.InvariantCulture)))}]");

        PreparedData data = DataPreparation.Prepare(spec, source, FeatureScaling.None, validationFraction: 0.1f);
        context.Text(
            $"validation_split=0.1 → train {data.TrainRowCount} / val {data.ValidationRowCount}. Файл отсортирован (сначала все R, потом все M), " +
            "поэтому строки перемешиваются: Keras взял бы последние 10% — одни «M». Признаки уже в [0, 1], нормализация не нужна. " +
            "Инициализация весов — Glorot uniform (умолчание Keras) вместо kernel_initializer='normal' из листинга.");

        context.Section("5–7. Базовая модель 60 → 60 → 1 (relu, sigmoid), adam, binary_crossentropy");
        var baseline = new ExperimentConfig("Базовая 60-60-1", new[] { 60 }, OptimizerKind.Adam, 0.001f, context.Epochs(100), BatchSize: 10);
        var results = new List<ExperimentResult> { ExperimentRunner.Run(context, data, baseline) };

        context.Section("8. Эксперимент 1: уменьшенный первый скрытый слой (60 → 30 → 1)");
        results.Add(ExperimentRunner.Run(context, data, baseline with { Name = "Эксперимент 1: 60-30-1", Hidden = new[] { 30 } }));

        context.Section("9. Эксперимент 2: дополнительный скрытый слой (60 → 60 → 15 → 1)");
        results.Add(ExperimentRunner.Run(context, data, baseline with { Name = "Эксперимент 2: 60-60-15-1", Hidden = new[] { 60, 15 } }));

        context.Section("10. Сравнение моделей и анализ результатов");
        context.Table(ExperimentAnalysis.Comparison(results));
        context.ChartLines("comparison-val-accuracy", "Sonar: валидационная точность", "val accuracy", ExperimentAnalysis.ValidationSeries(results));
        foreach (ExperimentResult result in results.Skip(1))
        {
            context.Bullet(ExperimentAnalysis.Versus(results[0], result));
        }

        context.Text("Переобучение (разрыв между loss и val_loss):");
        foreach (ExperimentResult result in results)
        {
            context.Bullet(ExperimentAnalysis.Overfitting(result));
        }

        context.Text(
            $"Валидационная выборка мала ({data.ValidationRowCount} шт.): один образец = {100f / data.ValidationRowCount:F1} п.п. точности, " +
            $"поэтому разница в 1–2 образца между моделями — шум. Ниже — более надёжная оценка {CrossValidationFolds}-fold перекрёстной проверкой.");

        context.Section($"Дополнительно: {CrossValidationFolds}-fold перекрёстная проверка тех же архитектур");
        Dataset shuffled = DatasetSplitter.Shuffle(all, DataPreparation.DefaultSplitSeed);
        var crossValidation = new TextTable("Модель", "Средняя val accuracy", "Разброс (σ) по блокам", "Блоки");
        CrossValidationResult? bestCv = null;
        float bestMean = float.MinValue;
        foreach (ExperimentResult result in results)
        {
            CrossValidationResult cv = CrossValidation.Run(context, spec, shuffled, result.Config, CrossValidationFolds, FeatureScaling.None);
            float[] finals = cv.FinalValidationMetrics();
            float mean = finals.Average();
            crossValidation.Add(
                result.Config.Name,
                mean.ToString("P1", CultureInfo.InvariantCulture),
                (CrossValidation.StandardDeviation(finals) * 100f).ToString("F1", CultureInfo.InvariantCulture) + " п.п.",
                string.Join(" ", finals.Select(f => f.ToString("P0", CultureInfo.InvariantCulture))));
            if (mean > bestMean)
            {
                bestMean = mean;
                bestCv = cv;
            }
        }

        context.Table(crossValidation);
        ExperimentResult bestSingle = ExperimentAnalysis.SelectBest(results);
        context.Text(
            $"**Вывод.** По одной валидационной выборке лучшая — «{bestSingle.Config.Name}» ({MetricFormat.Value(bestSingle.Metric, bestSingle.Validation.Metric)}); " +
            $"по перекрёстной проверке — «{bestCv!.Name}» (в среднем {bestMean:P1}). Перекрёстная оценка надёжнее: она усредняет {CrossValidationFolds} разбиений с валидацией по ~{all.RowCount / CrossValidationFolds} шт.");
        ModelCheckpoint.SaveAndVerify(context, bestSingle, "sonar_best.nn");
    }

    private static string FormatHead(Dataset dataset, int row) =>
        string.Join(", ", dataset.Row(row).Slice(0, 5).ToArray().Select(v => v.ToString("0.0000", CultureInfo.InvariantCulture)))
        + ", ... , " + SonarDataLoader.ClassNames[(int)dataset.Labels[row]];
}
