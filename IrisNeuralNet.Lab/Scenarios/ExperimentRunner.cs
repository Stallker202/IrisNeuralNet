using System;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Text;
using IrisNeuralNet.Core;
using IrisNeuralNet.Diagnostics;
using IrisNeuralNet.Lab.Experiments;
using IrisNeuralNet.Lab.Reporting;
using IrisNeuralNet.Optimizers;

namespace IrisNeuralNet.Lab.Scenarios;

/// <summary>Общий шаг всех сценариев: построить модель, вывести summary, обучить, оценить, сохранить график.</summary>
internal static class ExperimentRunner
{
    public static ExperimentResult Run(ScenarioContext context, PreparedData data, ExperimentConfig config, bool showSummary = true)
    {
        context.Text(
            $"**{config.Name}**: скрытые слои [{ArchitectureParser.Format(config.Hidden)}], " +
            $"optimizer='{OptimizerFactory.KerasName(config.Optimizer)}' (lr {config.LearningRate.ToString(CultureInfo.InvariantCulture)}), " +
            $"loss: {data.Task.Description}, epochs={config.Epochs}, batch_size={config.BatchSize}, " +
            $"признаки: {data.Scaling.DisplayName()}");

        var history = new TrainingHistory();
        var progress = new ConsoleProgress(context.Output, data.Task.Metric, config.Epochs);
        Experiment experiment = Experiment.Create(data, config, new CompositeTrainingObserver(history, progress));
        if (showSummary)
        {
            context.Code(ModelSummary.Format(experiment.Network, data.Spec.InputShape.ToArray()));
        }

        var stopwatch = Stopwatch.StartNew();
        experiment.Run();
        stopwatch.Stop();

        var result = new ExperimentResult(experiment, history, experiment.EvaluateTrain(), experiment.EvaluateValidation(), stopwatch.Elapsed);
        Metric metric = result.Metric;
        context.Text(
            $"evaluate (валидация/тест): loss {MetricFormat.Loss(result.Validation.Loss)}, {metric.Name} {MetricFormat.Value(metric, result.Validation.Metric)}; " +
            $"train: loss {MetricFormat.Loss(result.Train.Loss)}, {metric.Name} {MetricFormat.Value(metric, result.Train.Metric)}; " +
            $"время {Seconds(result.Elapsed)}");
        context.ChartHistory(Slug(config.Name), config.Name, history, MetricFormat.Label(metric));
        return result;
    }

    public static string Seconds(TimeSpan elapsed) =>
        elapsed.TotalSeconds.ToString(elapsed.TotalSeconds < 10 ? "F2" : "F1", CultureInfo.InvariantCulture) + " с";

    /// <summary>Имя файла из названия модели: транслит, латиница, цифры и дефисы.</summary>
    public static string Slug(string name)
    {
        var slug = new StringBuilder();
        foreach (char c in name.ToLowerInvariant())
        {
            string mapped = Transliterate(c);
            foreach (char m in mapped)
            {
                if (char.IsAsciiLetterOrDigit(m)) slug.Append(m);
                else if (slug.Length > 0 && slug[^1] != '-') slug.Append('-');
            }
        }

        return slug.ToString().Trim('-');
    }

    private static string Transliterate(char c)
    {
        const string cyrillic = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя";
        string[] latin =
        {
            "a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "i", "k", "l", "m", "n", "o", "p",
            "r", "s", "t", "u", "f", "h", "c", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya",
        };
        int index = cyrillic.IndexOf(c);
        return index >= 0 ? latin[index] : c.ToString();
    }
}
