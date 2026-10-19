using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using IrisNeuralNet.Data;
using IrisNeuralNet.Lab.Experiments;
using IrisNeuralNet.Lab.Reporting;
using IrisNeuralNet.Optimizers;

namespace IrisNeuralNet.Lab.Scenarios;

/// <summary>Лабораторная «Классификация цветков ириса с помощью ИНС в Keras» (многоклассовая классификация).</summary>
internal sealed class IrisScenario : ILabScenario
{
    public string Key => "iris";

    public string Title => "Лабораторная: классификация цветков ириса (Iris, многоклассовая классификация)";

    public void Run(ScenarioContext context)
    {
        DatasetSpec spec = Datasets.Iris;

        context.Section("1. Загрузка и анализ данных (листинги 1–3)");
        DatasetSource source = spec.Load();
        Dataset all = source.Train;
        context.Text($"Датасет: {all.RowCount} образцов, X: ({all.RowCount}, {all.FeatureCount}) — sepal/petal length/width в см, Y: ({all.RowCount},) — вид ириса.");
        context.Code(string.Join("\n", Enumerable.Range(0, 3).Select(i => FormatRow(all, i))));

        context.Section("2. Кодирование меток: LabelEncoder → one-hot (листинги 4–5)");
        var encoding = new TextTable("Исходная метка", "encoded_Y", "Вектор кодировки (dummy_y)", "Образцов");
        int[] classes = all.ClassIndices();
        for (int c = 0; c < IrisDataLoader.ClassCount; c++)
        {
            float[] oneHot = OneHotEncoder.Encode(new[] { c }, IrisDataLoader.ClassCount);
            encoding.Add(IrisDataLoader.ClassNames[c], c.ToString(CultureInfo.InvariantCulture),
                $"[{string.Join(", ", oneHot.Select(v => v.ToString(CultureInfo.InvariantCulture)))}]",
                classes.Count(x => x == c).ToString(CultureInfo.InvariantCulture));
        }

        context.Table(encoding);

        PreparedData raw = DataPreparation.Prepare(spec, source, FeatureScaling.None, validationFraction: 0.2f);
        context.Text(
            $"Разбиение: train {raw.TrainRowCount} / validation {raw.ValidationRowCount} (20%, seed {DataPreparation.DefaultSplitSeed}). " +
            "В Keras validation_split берёт последние 20% строк без перемешивания — в отсортированном iris.csv это были бы " +
            "только Iris-virginica, и val_accuracy ничего бы не говорила. Поэтому строки перемешиваются до разбиения.");
        context.Text($"dummy_y: ({raw.TrainRowCount}, 3); признаки подаются как есть, без нормализации — как в листинге.");

        context.Section("3. Базовая модель (листинги 6–7): Dense(8, relu) → Dense(3, softmax)");
        var baseline = new ExperimentConfig("Базовая 4-8-3", new[] { 8 }, OptimizerKind.Adam, 0.001f, context.Epochs(150), BatchSize: 10);
        var results = new List<ExperimentResult> { ExperimentRunner.Run(context, raw, baseline) };

        context.Section("4. Эксперименты: нейроны, слои, batch_size, оптимизаторы, эпохи, нормализация");
        PreparedData standardized = DataPreparation.Prepare(spec, source, FeatureScaling.Standardize, validationFraction: 0.2f);
        (PreparedData Data, ExperimentConfig Config)[] plan =
        {
            (raw, baseline with { Name = "4 нейрона (исходный листинг)", Hidden = new[] { 4 } }),
            (raw, baseline with { Name = "16 нейронов", Hidden = new[] { 16 } }),
            (raw, baseline with { Name = "Два слоя 8-8", Hidden = new[] { 8, 8 } }),
            (raw, baseline with { Name = "batch 5", BatchSize = 5 }),
            (raw, baseline with { Name = "batch 32", BatchSize = 32 }),
            (raw, baseline with { Name = "rmsprop", Optimizer = OptimizerKind.RmsProp }),
            (raw, baseline with { Name = "sgd", Optimizer = OptimizerKind.Sgd, LearningRate = OptimizerFactory.DefaultLearningRate(OptimizerKind.Sgd) }),
            (raw, baseline with { Name = RussianPlural.Epochs(context.Epochs(500)), Epochs = context.Epochs(500) }),
            (standardized, baseline with { Name = "z-score нормализация" }),
        };

        foreach ((PreparedData data, ExperimentConfig config) in plan)
        {
            results.Add(ExperimentRunner.Run(context, data, config, showSummary: false));
        }

        context.Section("5. Сравнение моделей и выбор лучшей (требование 4)");
        context.Table(ExperimentAnalysis.Comparison(results));
        context.ChartLines("comparison-val-accuracy", "Iris: валидационная точность моделей", "val accuracy", ExperimentAnalysis.ValidationSeries(results));

        ExperimentResult reference = results[0];
        foreach (ExperimentResult result in results.Skip(1))
        {
            context.Bullet(ExperimentAnalysis.Versus(reference, result));
        }

        context.Text("Переобучение:");
        foreach (ExperimentResult result in results)
        {
            context.Bullet(ExperimentAnalysis.Overfitting(result));
        }

        ExperimentResult best = ExperimentAnalysis.SelectBest(results);
        context.Text(
            $"**Лучшая модель: «{best.Config.Name}»** ({best.Architecture}) — val accuracy {MetricFormat.Value(best.Metric, best.Validation.Metric)}, " +
            $"val_loss {MetricFormat.Loss(best.Validation.Loss)}. Критерий — финальная валидационная точность; при равенстве " +
            "меньший val_loss (модель увереннее в верных ответах), затем меньше параметров. С валидацией в 30 образцов " +
            "многие модели дают 100% — различия в пределах 1 образца (3.3 п.п.) статистически незначимы.");
        ModelCheckpoint.SaveAndVerify(context, best, "iris_best.nn");
    }

    private static string FormatRow(Dataset dataset, int row) =>
        string.Join(", ", dataset.Row(row).ToArray().Select(v => v.ToString("0.0", CultureInfo.InvariantCulture)))
        + ", " + IrisDataLoader.ClassNames[(int)dataset.Labels[row]];
}
