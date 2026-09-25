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
        private readonly MetricsRing _ring = new();

        private CancellationTokenSource? _cancellation;
        private Task? _run;
        private AdamOptimizer? _optimizer;
        private float _learningRate = 0.02f;

        private NeuralNetwork? _network;
        public NeuralNetwork? CurrentNetwork => Volatile.Read(ref _network);

        public TrainingSession(
            float[] trainFeatures, float[] trainLabels,
            float[] validationFeatures, float[] validationLabels,
            int inputDim, int classCount)
        {
            _trainFeatures = trainFeatures;
            _trainLabels = trainLabels;
            _validationFeatures = validationFeatures;
            _validationLabels = validationLabels;
            _inputDim = inputDim;
            _classCount = classCount;
        }

        public MetricsRing Ring => _ring;
        public bool IsRunning => _run is { IsCompleted: false };

        public float LearningRate
        {
            get => _learningRate;
            set
            {
                _learningRate = value;
                AdamOptimizer? optimizer = _optimizer;
                if (optimizer is not null)
                {
                    optimizer.LearningRate = value;   // подхватится на следующем шаге
                }
            }
        }

        public void Start(int[] hiddenNeurons, int epochs, int batchSize, int seed)
        {
            Stop();
            _ring.Clear();
            _cancellation = new CancellationTokenSource();
            CancellationToken token = _cancellation.Token;
            int[] architecture = (int[])hiddenNeurons.Clone();
            _run = Task.Run(() => RunLoop(architecture, epochs, batchSize, seed, token), token);
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

        private void RunLoop(int[] hiddenNeurons, int epochs, int batchSize, int seed, CancellationToken token)
        {
            try
            {
                NeuralNetwork network = BuildNetwork(hiddenNeurons, seed);
                Volatile.Write(ref _network, network);
                AdamOptimizer optimizer = new(network, _learningRate);
                _optimizer = optimizer;
                Trainer trainer = new(network, optimizer, new CategoricalCrossEntropyLoss(), _ring);
                trainer.Fit(_trainFeatures, _trainLabels, _validationFeatures, _validationLabels,
                    epochs, batchSize, seed, token);
            }
            catch (OperationCanceledException)
            {
                // Остановлено пользователем.
            }
            finally
            {
                _optimizer = null;
            }
        }

        private NeuralNetwork BuildNetwork(int[] hiddenNeurons, int seed)
        {
            var layers = new List<DenseLayer>();
            int input = _inputDim;
            for (int i = 0; i < hiddenNeurons.Length; i++)
            {
                layers.Add(new DenseLayer(input, hiddenNeurons[i], new ReLUActivation(), seed: seed + i + 1));
                input = hiddenNeurons[i];
            }

            layers.Add(new DenseLayer(input, _classCount, new SoftmaxActivation(), seed: seed + hiddenNeurons.Length + 1));
            return new NeuralNetwork(layers.ToArray());
        }
    }
}
