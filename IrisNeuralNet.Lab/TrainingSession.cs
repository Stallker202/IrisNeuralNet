using System;
using System.Threading;
using System.Threading.Tasks;
using IrisNeuralNet.Core;
using IrisNeuralNet.Diagnostics;
using IrisNeuralNet.Lab.Experiments;
using IrisNeuralNet.Optimizers;

namespace IrisNeuralNet.Lab;

/// <summary>
/// Фоновый прогон обучения. UI-поток не блокирует обучение, обучение не ждёт UI:
/// метрики текут через MetricsRing, learning rate — атомарной записью в оптимизатор.
/// Stop() дожидается завершения: отмена проверяется перед каждым батчем, поэтому ждать недолго,
/// а старый прогон гарантированно не допишет метрики в кольцо нового.
/// </summary>
internal sealed class TrainingSession : IDisposable
{
    private readonly MetricsRing _ring = new(capacity: 8192);
    private CancellationTokenSource? _cancellation;
    private Task? _run;
    private IOptimizer? _optimizer;
    private NeuralNetwork? _network;
    private Exception? _error;
    private float _learningRate;

    public TrainingSession(PreparedData data, float learningRate)
    {
        Data = data ?? throw new ArgumentNullException(nameof(data));
        _learningRate = learningRate;
    }

    public PreparedData Data { get; }
    public MetricsRing Ring => _ring;
    public bool IsRunning => _run is { IsCompleted: false };
    public NeuralNetwork? CurrentNetwork => Volatile.Read(ref _network);

    /// <summary>Исключение последнего прогона (кроме отмены), чтобы UI мог показать причину.</summary>
    public Exception? LastError => Volatile.Read(ref _error);

    public float LearningRate
    {
        get => _learningRate;
        set
        {
            _learningRate = value;
            IOptimizer? optimizer = Volatile.Read(ref _optimizer);
            if (optimizer is not null)
            {
                optimizer.LearningRate = value;
            }
        }
    }

    public void Start(ExperimentConfig config)
    {
        Stop();
        _ring.Clear();
        Volatile.Write(ref _error, null);
        _cancellation = new CancellationTokenSource();
        CancellationToken token = _cancellation.Token;
        ExperimentConfig effective = config with { LearningRate = _learningRate };
        _run = Task.Run(() => RunLoop(effective, token), token);
    }

    public void Stop()
    {
        CancellationTokenSource? cancellation = _cancellation;
        if (cancellation is null) return;

        cancellation.Cancel();
        try
        {
            _run?.Wait();
        }
        catch (AggregateException)
        {
            // Отмена до старта задачи — штатный путь, не ошибка.
        }

        cancellation.Dispose();
        _cancellation = null;
    }

    public void Dispose() => Stop();

    private void RunLoop(ExperimentConfig config, CancellationToken token)
    {
        try
        {
            Experiment experiment = Experiment.Create(Data, config, _ring);
            Volatile.Write(ref _network, experiment.Network);
            Volatile.Write(ref _optimizer, experiment.Optimizer);
            experiment.Run(token);
        }
        catch (OperationCanceledException)
        {
            // Остановлено пользователем.
        }
        catch (Exception e)
        {
            Volatile.Write(ref _error, e);
        }
        finally
        {
            Volatile.Write(ref _optimizer, null);
        }
    }
}
