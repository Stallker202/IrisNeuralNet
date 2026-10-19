using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using IrisNeuralNet.Core;
using IrisNeuralNet.Data;
using IrisNeuralNet.Lab.Experiments;
using IrisNeuralNet.Lab.Reporting;
using IrisNeuralNet.Optimizers;

namespace IrisNeuralNet.Lab.Scenarios;

/// <summary>Лабораторная «Классификация изображений MNIST (рукописные цифры)».</summary>
internal sealed class MnistScenario : ILabScenario
{
    private const float TargetAccuracy = 0.95f;
    private const int Side = MnistDataLoader.ImageSide;
    private const int GridSamples = 16;

    public string Key => "mnist";

    public string Title => "Лабораторная: классификация изображений MNIST (рукописные цифры)";

    public void Run(ScenarioContext context)
    {
        DatasetSpec spec = Datasets.Mnist;

        context.Section("3. Загрузка и анализ данных");
        var progress = new Progress<string>(message => context.Output.WriteLine("  " + message));
        string directory = MnistDownloader.EnsureAsync(progress, default).GetAwaiter().GetResult();
        DatasetSource source = spec.Load();
        Dataset train = source.Train;
        Dataset test = source.Test!.Value;
        context.Text($"Файлы: {directory}");
        context.Text($"Форма обучающих изображений: ({train.RowCount}, {Side}, {Side}), меток: ({train.RowCount},); " +
                     $"тестовых: ({test.RowCount}, {Side}, {Side}) и ({test.RowCount},).");
        context.Text($"Пример меток: [{string.Join(" ", train.Labels.Slice(0, 10).ToArray())}]");
        context.Code($"Пример изображения: {train.Labels[0]}\n{Ascii(train.Row(0))}");
        context.ChartDigits("train-samples", "Первые изображения обучающей выборки", Samples(train, Enumerable.Range(0, GridSamples), i => $"метка {train.Labels[i]}"));

        context.Section("4. Предобработка данных");
        PreparedData data = DataPreparation.Prepare(spec, source, FeatureScaling.Pixels, validationFraction: 0f);
        context.Text($"a) Flatten: 28×28 → вектор {data.FeatureCount}; нормализация ÷255 → диапазон значений пикселей " +
                     $"[{data.TrainFeatures.Min().ToString(CultureInfo.InvariantCulture)}, {data.TrainFeatures.Max().ToString(CultureInfo.InvariantCulture)}].");
        context.Text($"b) One-hot: обучение ({data.TrainRowCount}, 10), тест ({data.ValidationRowCount}, 10); метка {train.Labels[0]} → " +
                     $"[{string.Join(", ", data.TrainTargets.AsSpan(0, 10).ToArray())}].");

        context.Section("5–7. Базовая модель: Flatten → Dense(256, relu) → Dense(10, softmax), adam");
        int baseEpochs = context.Epochs(5);
        int longEpochs = context.Epochs(20);
        var baseline = new ExperimentConfig($"Базовая модель ({RussianPlural.Epochs(baseEpochs)})", new[] { 256 }, OptimizerKind.Adam, 0.001f, baseEpochs, BatchSize: 128);
        var results = new List<ExperimentResult> { ExperimentRunner.Run(context, data, baseline) };

        context.Section("8. Эксперимент: поиск модели с точностью ≥ 95%");
        var improved = baseline with { Name = $"Улучшенная модель ({RussianPlural.Epochs(longEpochs)})", Hidden = new[] { 512, 256 }, Epochs = longEpochs };
        ExperimentResult improvedResult = ExperimentRunner.Run(context, data, improved);
        results.Add(improvedResult);
        context.Text(improvedResult.Validation.Metric >= TargetAccuracy
            ? $"Цель достигнута: точность {improvedResult.Validation.Metric:P2} ≥ 95%."
            : $"Цель не достигнута. Текущая точность: {improvedResult.Validation.Metric:P2}. Продолжайте экспериментировать!");

        context.Section("9. Эксперимент: другой оптимизатор (RMSprop)");
        ExperimentResult rmsprop = ExperimentRunner.Run(context, data, improved with { Name = $"Модель (RMSprop, {RussianPlural.Epochs(longEpochs)})", Optimizer = OptimizerKind.RmsProp }, showSummary: false);
        results.Add(rmsprop);
        context.Text($"Сравнение точностей: улучшенная модель (Adam): {improvedResult.Validation.Metric:P2}; модель (RMSprop): {rmsprop.Validation.Metric:P2}.");

        ExperimentResult best = ExperimentAnalysis.SelectBest(results);

        context.Section("10. Классификация пользовательского изображения (демонстрация)");
        ClassifySample(context, best, test, index: 0);
        if (context.Options.ImagePath is { } imagePath)
        {
            ClassifyFile(context, best, imagePath);
        }

        context.Section("11. Сравнение и анализ результатов");
        context.Table(ExperimentAnalysis.Comparison(results));
        context.ChartLines("comparison-val-accuracy", "MNIST: точность на тесте по эпохам", "val accuracy", ExperimentAnalysis.ValidationSeries(results));
        context.Bullet($"Эпохи и архитектура: {ExperimentAnalysis.Versus(results[0], improvedResult)}");
        context.Bullet($"Adam vs RMSprop: {ExperimentAnalysis.Versus(improvedResult, rmsprop)}");
        foreach (ExperimentResult result in results)
        {
            context.Bullet(ExperimentAnalysis.Overfitting(result));
        }

        ConfusionMatrix(context, best, test);
        ModelCheckpoint.SaveAndVerify(context, best, "mnist_best.nn");
    }

    /// <summary>classify_custom_image: батч из одного изображения → вероятности → argmax и уверенность.</summary>
    private static void ClassifySample(ScenarioContext context, ExperimentResult model, Dataset test, int index)
    {
        float[] probabilities = Predict(model, test.Row(index).ToArray());
        int predicted = Metrics.ArgMax(probabilities);
        context.Text($"Тестовое изображение #{index}: истинная метка {test.Labels[index]}, предсказанная {predicted}, " +
                     $"уверенность {probabilities[predicted]:P2} (модель «{model.Config.Name}»).");
        context.Code(Ascii(test.Row(index)));
    }

    private static void ClassifyFile(ScenarioContext context, ExperimentResult model, string path)
    {
        if (context.Options.Images is null)
        {
            context.Text($"Загрузка изображений недоступна на этой платформе — файл {path} пропущен.");
            return;
        }

        float[] grayscale = context.Options.Images.LoadGrayscale(path, out int width, out int height);
        DigitImagePreprocessor.ToInk(grayscale, width, height);
        float[] pixels = DigitImagePreprocessor.ToMnist(grayscale, width, height);
        float[] probabilities = Predict(model, pixels);
        string top3 = string.Join(", ", probabilities
            .Select((p, digit) => (p, digit))
            .OrderByDescending(x => x.p)
            .Take(3)
            .Select(x => $"{x.digit}: {x.p:P1}"));
        context.Text($"Файл {path} ({width}×{height}) → приведён к виду MNIST (инверсия фона, рамка 20×20, центр масс в центре 28×28). Топ-3: {top3}.");
        context.Code(Ascii(pixels));
        context.ChartDigits("custom-image", "Пользовательское изображение после предобработки",
            new[] { new DigitSample(pixels, $"предсказано {Metrics.ArgMax(probabilities)}") });
    }

    private static void ConfusionMatrix(ScenarioContext context, ExperimentResult model, Dataset test)
    {
        PreparedData data = model.Experiment.Data;
        int classes = MnistDataLoader.ClassCount;
        float[] output = new float[test.RowCount * classes];
        model.Experiment.Network.Infer(data.ValidationFeatures, output, test.RowCount);

        var counts = new int[classes, classes];
        var errors = new List<int>();
        for (int i = 0; i < test.RowCount; i++)
        {
            int actual = (int)test.Labels[i];
            int predicted = Metrics.ArgMax(output.AsSpan(i * classes, classes));
            counts[actual, predicted]++;
            if (actual != predicted) errors.Add(i);
        }

        var headers = new[] { "истина \\ предсказано" }
            .Concat(Enumerable.Range(0, classes).Select(c => c.ToString(CultureInfo.InvariantCulture)))
            .Append("точность класса")
            .ToArray();
        var table = new TextTable(headers);
        for (int actual = 0; actual < classes; actual++)
        {
            int total = Enumerable.Range(0, classes).Sum(p => counts[actual, p]);
            var row = new List<string> { actual.ToString(CultureInfo.InvariantCulture) };
            row.AddRange(Enumerable.Range(0, classes).Select(p => counts[actual, p].ToString(CultureInfo.InvariantCulture)));
            row.Add(((float)counts[actual, actual] / total).ToString("P1", CultureInfo.InvariantCulture));
            table.Add(row.ToArray());
        }

        context.Text($"Матрица ошибок лучшей модели «{model.Config.Name}» на {test.RowCount} тестовых изображениях (ошибок: {errors.Count}):");
        context.Table(table);
        context.ChartDigits("misclassified", "Ошибки лучшей модели: истина → предсказание",
            Samples(test, errors.Take(GridSamples), i => $"{test.Labels[i]} → {Metrics.ArgMax(output.AsSpan(i * classes, classes))}"));
    }

    private static float[] Predict(ExperimentResult model, float[] rawPixels)
    {
        PreparedData data = model.Experiment.Data;
        float[] features = data.Normalizer?.Transform(rawPixels, rowCount: 1) ?? rawPixels;
        float[] probabilities = new float[MnistDataLoader.ClassCount];
        model.Experiment.Network.Predict(features, probabilities);
        return probabilities;
    }

    private static DigitSample[] Samples(Dataset dataset, IEnumerable<int> indices, Func<int, string> caption) =>
        indices.Select(i => new DigitSample(dataset.Row(i).ToArray(), caption(i))).ToArray();

    /// <summary>ASCII-рендер 28×28 для консоли (аналог plt.imshow с cmap=binary).</summary>
    private static string Ascii(ReadOnlySpan<float> pixels)
    {
        const string ramp = " .:-=+*#%@";
        var text = new StringBuilder();
        for (int y = 0; y < Side; y++)
        {
            for (int x = 0; x < Side; x++)
            {
                int level = (int)(Math.Clamp(pixels[y * Side + x], 0f, 255f) / 256f * ramp.Length);
                text.Append(ramp[level]).Append(ramp[level]);
            }

            if (y < Side - 1) text.Append('\n');
        }

        return text.ToString();
    }
}
