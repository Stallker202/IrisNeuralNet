using System;
using System.Threading;
using IrisNeuralNet.Diagnostics;
using IrisNeuralNet.MathCore;
using IrisNeuralNet.Optimizers;

namespace IrisNeuralNet.Core;

/// <summary>
/// Mini-batch тренер (аналог model.fit / model.evaluate): перемешивает выборку,
/// гоняет forward → loss → backward → step, раз в эпоху считает потери и метрику.
/// Все буферы grow-only: после прогрева цикл обучения не аллоцирует.
/// Оценка идёт кусками по <see cref="EvaluationChunkRows"/> строк — память не растёт с размером выборки.
/// Отмена кооперативная: токен проверяется перед каждым батчем и каждым куском оценки.
/// </summary>
public sealed class Trainer
{
    /// <summary>Размер куска при оценке больших выборок (MNIST: 60000 строк).</summary>
    public const int EvaluationChunkRows = 1024;

    private readonly NeuralNetwork _network;
    private readonly IOptimizer _optimizer;
    private readonly ILossFunction _loss;
    private readonly ITrainingObserver _observer;
    private float[] _batchX = Array.Empty<float>();
    private float[] _batchY = Array.Empty<float>();
    private float[] _predicted = Array.Empty<float>();
    private float[] _outputGradient = Array.Empty<float>();
    private int[] _order = Array.Empty<int>();

    /// <summary>Функция потерь и метрика берутся из типа задачи.</summary>
    public Trainer(NeuralNetwork network, IOptimizer optimizer, TrainingTask task, ITrainingObserver? observer = null)
        : this(network, optimizer, (task ?? throw new ArgumentNullException(nameof(task))).CreateLoss(), task.Metric, observer)
    {
    }

    public Trainer(NeuralNetwork network, IOptimizer optimizer, ILossFunction loss, Metric metric, ITrainingObserver? observer = null)
    {
        _network = network ?? throw new ArgumentNullException(nameof(network));
        _optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));
        _loss = loss ?? throw new ArgumentNullException(nameof(loss));
        Metric = metric ?? throw new ArgumentNullException(nameof(metric));
        _observer = observer ?? NullTrainingObserver.Instance;
    }

    public Metric Metric { get; }

    /// <summary>Обучение без валидационной выборки. Возвращает метрики последней эпохи.</summary>
    public EpochMetrics Fit(
        ReadOnlySpan<float> trainX,
        ReadOnlySpan<float> trainY,
        in TrainingOptions options,
        CancellationToken cancellationToken = default) =>
        Fit(trainX, trainY, ReadOnlySpan<float>.Empty, ReadOnlySpan<float>.Empty, options, cancellationToken);

    /// <summary>
    /// Обучение с валидацией (аналог validation_data). Пустая валидация — «без валидации».
    /// Возвращает метрики последней выполненной эпохи (с учётом ранней остановки).
    /// </summary>
    public EpochMetrics Fit(
        ReadOnlySpan<float> trainX,
        ReadOnlySpan<float> trainY,
        ReadOnlySpan<float> validationX,
        ReadOnlySpan<float> validationY,
        in TrainingOptions options,
        CancellationToken cancellationToken = default)
    {
        options.Validate();
        int rowCount = EnsureDataset(trainX, trainY);
        bool hasValidation = !validationX.IsEmpty || !validationY.IsEmpty;
        if (hasValidation)
        {
            EnsureDataset(validationX, validationY);
        }

        int[] order = ResetOrder(rowCount);
        var random = new Random(options.Seed);
        float bestValidationLoss = float.MaxValue;
        int epochsWithoutImprovement = 0;
        EpochMetrics metrics = default;

        for (int epoch = 1; epoch <= options.Epochs; epoch++)
        {
            Shuffler.Shuffle(order.AsSpan(0, rowCount), random);
            for (int start = 0; start < rowCount; start += options.BatchSize)
            {
                cancellationToken.ThrowIfCancellationRequested();
                TrainBatch(trainX, trainY, order, start, Math.Min(options.BatchSize, rowCount - start));
            }

            Evaluation train = Evaluate(trainX, trainY, cancellationToken);
            Evaluation validation = hasValidation ? Evaluate(validationX, validationY, cancellationToken) : Evaluation.None;
            metrics = new EpochMetrics(epoch, train.Loss, train.Metric, validation.Loss, validation.Metric);
            _observer.OnEpochCompleted(in metrics);

            if (hasValidation && options.EarlyStoppingPatience > 0)
            {
                if (validation.Loss < bestValidationLoss)
                {
                    bestValidationLoss = validation.Loss;
                    epochsWithoutImprovement = 0;
                }
                else if (++epochsWithoutImprovement >= options.EarlyStoppingPatience)
                {
                    break;
                }
            }
        }

        return metrics;
    }

    /// <summary>Потери и метрика на выборке (аналог model.evaluate), кусками фиксированного размера.</summary>
    public Evaluation Evaluate(ReadOnlySpan<float> x, ReadOnlySpan<float> y, CancellationToken cancellationToken = default)
    {
        int rowCount = EnsureDataset(x, y);
        int inputWidth = _network.InputDim;
        int outputWidth = _network.OutputDim;

        // double-аккумуляторы: сумма тысяч кусков не теряет точность float.
        double lossSum = 0;
        double metricSum = 0;
        for (int start = 0; start < rowCount; start += EvaluationChunkRows)
        {
            cancellationToken.ThrowIfCancellationRequested();

            int count = Math.Min(EvaluationChunkRows, rowCount - start);
            ReadOnlySpan<float> chunkY = y.Slice(start * outputWidth, count * outputWidth);
            Span<float> predicted = Buffers.Rent(ref _predicted, count * outputWidth);
            _network.Infer(x.Slice(start * inputWidth, count * inputWidth), predicted, count);

            // Потери и метрики — средние по строкам, поэтому куски взвешиваются числом строк.
            lossSum += (double)_loss.Compute(predicted, chunkY, outputWidth) * count;
            metricSum += (double)Metric.Compute(predicted, chunkY, outputWidth) * count;
        }

        return new Evaluation((float)(lossSum / rowCount), (float)(metricSum / rowCount));
    }

    private void TrainBatch(ReadOnlySpan<float> trainX, ReadOnlySpan<float> trainY, int[] order, int start, int count)
    {
        int inputWidth = _network.InputDim;
        int outputWidth = _network.OutputDim;

        Span<float> batchX = Buffers.Rent(ref _batchX, count * inputWidth);
        Span<float> batchY = Buffers.Rent(ref _batchY, count * outputWidth);
        for (int i = 0; i < count; i++)
        {
            int row = order[start + i];
            trainX.Slice(row * inputWidth, inputWidth).CopyTo(batchX.Slice(i * inputWidth));
            trainY.Slice(row * outputWidth, outputWidth).CopyTo(batchY.Slice(i * outputWidth));
        }

        Span<float> predicted = Buffers.Rent(ref _predicted, count * outputWidth);
        _network.Forward(batchX, predicted, count);

        Span<float> outputGradient = Buffers.Rent(ref _outputGradient, count * outputWidth);
        _loss.Backward(predicted, batchY, outputWidth, outputGradient);
        _network.Backward(outputGradient, count);
        _optimizer.Step(_network);
    }

    /// <summary>Проверяет согласованность X и Y и возвращает число строк.</summary>
    private int EnsureDataset(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        SpanGuard.EnsureDivisible(x.Length, _network.InputDim, nameof(x));
        SpanGuard.EnsureDivisible(y.Length, _network.OutputDim, nameof(y));

        int rowCount = x.Length / _network.InputDim;
        if (rowCount != y.Length / _network.OutputDim)
        {
            throw new ArgumentException("X and Y must have the same row count.");
        }

        if (rowCount == 0)
        {
            throw new ArgumentException("Dataset must contain at least one row.");
        }

        return rowCount;
    }

    private int[] ResetOrder(int rowCount)
    {
        if (_order.Length < rowCount)
        {
            _order = new int[rowCount];
        }

        for (int i = 0; i < rowCount; i++)
        {
            _order[i] = i;
        }

        return _order;
    }
}
