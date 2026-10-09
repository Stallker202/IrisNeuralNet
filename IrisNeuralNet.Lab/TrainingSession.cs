using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using IrisNeuralNet.Core;
using IrisNeuralNet.Diagnostics;
using IrisNeuralNet.Optimizers;

/// Фоновый прогон обучения. UI-поток не блокирует обучение, обучение не ждёт UI:
/// метрики текут через MetricsRing, learning rate — атомарной записью в оптимизатор.

namespace IrisNeuralNet.Lab
{
    internal sealed class TrainingSession : IDisposable
    {
        private readonly float[] _trainFeatures;
        private readonly float[] _trainLabels;
        private readonly float[] _validationFeatures;
        private readonly float[] _validationLabels;
        private readonly int _inputDim;
        private readonly int _classCount;
        private readonly TaskKind _taskKind;
        private readonly MetricsRing _ring = new();

        private CancellationTokenSource? _cancellation;
        private Task? _run;
        private AdamOptimizer? _optimizer;
        private NeuralNetwork? _network;
        private float _learningRate = 0.02f;

        public TrainingSession(PreparedData data)
        {
            if (data is null) throw new ArgumentNullException(nameof(data));
            _trainFeatures = data.TrainFeatures;
            _trainLabels = data.TrainLabels;
            _validationFeatures = data.ValidationFeatures;
            _validationLabels = data.ValidationLabels;
            _inputDim = data.FeatureCount;
            _classCount = data.ClassCount;
            _taskKind = data.TaskKind;
        }

        public MetricsRing Ring => _ring;
        public bool IsRunning => _run is { IsCompleted: false };
        public NeuralNetwork? CurrentNetwork => Volatile.Read(ref _network);

        public float LearningRate
        {
            get => _learningRate;
            set
            {
                _learningRate = value;
                AdamOptimizer? optimizer = _optimizer;
                if (optimizer is not null)
                {
                    optimizer.LearningRate = value;
                }
            }
        }

        public void Start(int[] hiddenNeurons, int epochs, int batchSize, int seed, int earlyStoppingPatience = 0)
        {
            Stop();
            _ring.Clear();
            _cancellation = new CancellationTokenSource();
            CancellationToken token = _cancellation.Token;
            int[] architecture = (int[])hiddenNeurons.Clone();
            _run = Task.Run(
                () => RunLoop(architecture, epochs, batchSize, seed, earlyStoppingPatience, token),
                token);
        }

        public void Stop()
        {
            _cancellation?.Cancel();
            try
            {
                _run?.Wait(TimeSpan.FromSeconds(2));
            }
            catch (AggregateException)
            {
                // Отмена — нормальный путь, не ошибка.
            }

            _cancellation?.Dispose();
            _cancellation = null;
        }

        public void Dispose() => Stop();

        private void RunLoop(
            int[] hiddenNeurons, int epochs, int batchSize, int seed, int earlyStoppingPatience, CancellationToken token)
        {
            try
            {
                NeuralNetwork network = _taskKind == TaskKind.Binary
                    ? NetworkFactory.Create(_inputDim, hiddenNeurons, 1, new SigmoidActivation(), seed)
                    : NetworkFactory.Create(_inputDim, hiddenNeurons, _classCount, seed);
                Volatile.Write(ref _network, network);

                AdamOptimizer optimizer = new(network, _learningRate);
                _optimizer = optimizer;

                ILossFunction loss = _taskKind == TaskKind.Binary
                    ? new BinaryCrossEntropyLoss()
                    : new CategoricalCrossEntropyLoss();

                Trainer trainer = new(network, optimizer, loss, _ring, _taskKind);
                trainer.Fit(_trainFeatures, _trainLabels, _validationFeatures, _validationLabels,
                    epochs, batchSize, seed, token, earlyStoppingPatience);
            }
            catch (OperationCanceledException)
            {
                // Остановлено пользователем или early stopping-отменой.
            }
            finally
            {
                _optimizer = null;
            }
        }
    }
}
