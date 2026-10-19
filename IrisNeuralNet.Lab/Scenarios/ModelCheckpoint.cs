using System.IO;
using IrisNeuralNet.Core;
using IrisNeuralNet.Data;
using IrisNeuralNet.Lab.Experiments;
using IrisNeuralNet.Lab.Reporting;

namespace IrisNeuralNet.Lab.Scenarios;

/// <summary>Сохранение лучшей модели и проверка, что загруженная копия на сырых признаках даёт ту же метрику.</summary>
internal static class ModelCheckpoint
{
    public static void SaveAndVerify(ScenarioContext context, ExperimentResult result, string fileName)
    {
        PreparedData data = result.Experiment.Data;
        string path = context.PathOf(fileName);
        ModelFile.Save(result.Experiment.Network, data.Normalizer, path);
        LoadedModel loaded = ModelFile.Load(path);

        // Загруженная модель сама несёт нормализатор: подаём сырые признаки, как в реальном инференсе.
        Dataset raw = data.RawValidation;
        float[] features = loaded.Normalizer?.Transform(raw.Features, raw.RowCount) ?? raw.Features.ToArray();
        float[] predicted = new float[raw.RowCount * loaded.Network.OutputDim];
        loaded.Network.Infer(features, predicted, raw.RowCount);
        float metric = data.Task.Metric.Compute(predicted, data.ValidationTargets, loaded.Network.OutputDim);

        bool matches = System.MathF.Abs(metric - result.Validation.Metric) < 1e-5f;
        context.Text(
            $"Модель «{result.Config.Name}» сохранена в {fileName} ({new FileInfo(path).Length:N0} байт, нормализатор внутри) и загружена заново: " +
            $"val {data.Task.Metric.Name} на сырых признаках = {MetricFormat.Value(data.Task.Metric, metric)} " +
            (matches ? "— совпадает с исходной моделью." : "— НЕ совпадает с исходной моделью!"));
    }
}
