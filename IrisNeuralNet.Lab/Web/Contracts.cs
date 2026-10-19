using IrisNeuralNet.Lab.Experiments;
using IrisNeuralNet.Optimizers;

namespace IrisNeuralNet.Lab.Web;

/// <summary>Подготовить данные выбранного датасета (смена датасета, предобработки или доли валидации).</summary>
internal sealed record DataRequest(string Dataset, FeatureScaling Scaling, float ValidationFraction, float LearningRate);

/// <summary>Настройки прогона из левой панели (аналог compile + fit).</summary>
/// <param name="Hidden">Скрытые слои строкой: «512, 256»; «-» или пусто — без скрытых слоёв.</param>
/// <param name="KeepPrevious">Сохранить текущие кривые как «прошлый прогон» для оверлея.</param>
internal sealed record TrainRequest(
    string Hidden,
    OptimizerKind Optimizer,
    float LearningRate,
    int Epochs,
    int BatchSize,
    int EarlyStoppingPatience,
    bool KeepPrevious);

/// <summary>Sweep: оптимизатор, learning rate и batch — из шаблона, архитектура — из сетки.</summary>
internal sealed record SweepRequest(TrainRequest Template, int Epochs);

internal sealed record LearningRateRequest(float Value);

internal sealed record ErrorResponse(string Error);

internal sealed record CatalogDto(DatasetDto[] Datasets, OptionDto[] Scalings, OptimizerDto[] Optimizers);

internal sealed record DatasetDto(string Key, string Name, string Task, bool HasPredefinedTestSplit, DefaultsDto Defaults);

internal sealed record DefaultsDto(
    string Hidden,
    FeatureScaling Scaling,
    float ValidationFraction,
    OptimizerKind Optimizer,
    float LearningRate,
    int Epochs,
    int BatchSize);

internal sealed record OptionDto(string Value, string Label);

internal sealed record OptimizerDto(OptimizerKind Value, string Label, float DefaultLearningRate);

/// <param name="Revision">Растёт при каждом изменении, видимом на графиках: странице не нужно перерисовывать их зря.</param>
internal sealed record StateDto(
    int DataVersion,
    string? Dataset,
    bool Ready,
    bool Running,
    long Revision,
    string Status,
    float LearningRate,
    bool SupportsProjection,
    bool SupportsDigits,
    SweepStateDto Sweep);

internal sealed record SweepStateDto(bool Running, int Done, int Total, string Status, long Revision);

internal sealed record ArchitectureDto(bool Valid, string Error, int[] Hidden);

/// <param name="Preview">PNG 28×28 (data URI) — то, что видит сеть.</param>
/// <param name="Pixels">Те же 784 пикселя в base64: страница присылает их обратно для пересчёта во время обучения.</param>
/// <param name="Truth">Откуда изображение (истинная метка тестового образца, имя файла) — страница повторяет его при пересчёте.</param>
/// <param name="Bars">SVG со столбиками вероятностей.</param>
internal sealed record DigitResultDto(
    string Preview,
    string Pixels,
    float[] Probabilities,
    int Digit,
    string Verdict,
    string Truth,
    string Bars);
