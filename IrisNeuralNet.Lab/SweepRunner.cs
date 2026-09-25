using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using IrisNeuralNet.Core;
using IrisNeuralNet.Optimizers;

namespace IrisNeuralNet.Lab
{
    internal sealed class SweepConfig
    {
        public int[] NeuronOptions { get; init; } = { 2, 4, 8, 16, 32 };
        public int[] LayerOptions { get; init; } = { 0, 1, 2 };
        public int Epochs { get; init; } = 150;
        public int BatchSize { get; init; } = 16;
        public int BaseSeed { get; init; } = 100;
        public float LearningRate { get; init; } = 0.02f;
    }

    internal readonly struct SweepResult
    {
        public SweepResult(
            int hiddenLayers, int neurons,
            float trainAccuracy, float validationAccuracy, float finalLoss, long elapsedMs)
        {
            HiddenLayers = hiddenLayers;
            Neurons = neurons;
            TrainAccuracy = trainAccuracy;
            ValidationAccuracy = validationAccuracy;
            FinalLoss = finalLoss;
            ElapsedMs = elapsedMs;
        }

        public int HiddenLayers { get; }
        public int Neurons { get; }
        public float TrainAccuracy { get; }
        public float ValidationAccuracy { get; }
        public float FinalLoss { get; }
        public long ElapsedMs { get; }
    }

    /// <summary>
    /// Пакетное сравнение архитектур: каждый конфиг — независимый прогон,
    /// поэтому sweep распараллелен без разделяемого состояния (кроме потокобезопасного мешка результатов).
    /// </summary>
    internal sealed class SweepRunner
    {
        private readonly float[] _trainFeatures;
        private readonly float[] _trainLabels;
        private readonly float[] _validationFeatures;
        private readonly float[] _validationLabels;
        private readonly int _inputDim;
        private readonly int _classCount;

        public SweepRunner(
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

        public List<SweepResult> Run(SweepConfig config, IProgress<SweepResult>? progress = null, CancellationToken token = default)
        {
            if (config is null) throw new ArgumentNullException(nameof(config));

            var pairs = new List<(int Layers, int Neurons, int Index)>();
            int index = 0;
            foreach (int layers in config.LayerOptions)
            {
                foreach (int neurons in config.NeuronOptions)
                {
                    pairs.Add((layers, neurons, index));
                    index++;
                }
            }

            var bag = new ConcurrentBag<SweepResult>();
            Parallel.ForEach(pairs, new ParallelOptions { CancellationToken = token }, pair =>
            {
                var stopwatch = Stopwatch.StartNew();
                int seed = config.BaseSeed + pair.Index;
                int[] hidden = BuildHidden(pair.Layers, pair.Neurons);

                NeuralNetwork network = NetworkFactory.Create(_inputDim, hidden, _classCount, seed);
                var optimizer = new AdamOptimizer(network, config.LearningRate);
                var trainer = new Trainer(network, optimizer, new CategoricalCrossEntropyLoss());

                float loss = trainer.Fit(
                    _trainFeatures, _trainLabels,
                    _validationFeatures, _validationLabels,
                    config.Epochs, config.BatchSize, seed, token);
                stopwatch.Stop();

                var result = new SweepResult(
                    pair.Layers, pair.Neurons,
                    trainer.EvaluateAccuracy(_trainFeatures, _trainLabels),
                    trainer.EvaluateAccuracy(_validationFeatures, _validationLabels),
                    loss,
                    stopwatch.ElapsedMilliseconds);
                bag.Add(result);
                progress?.Report(result);
            });

            return bag.OrderBy(r => r.HiddenLayers).ThenBy(r => r.Neurons).ToList();
        }

        private static int[] BuildHidden(int layers, int neurons)
        {
            var hidden = new int[layers];
            for (int i = 0; i < layers; i++)
            {
                hidden[i] = neurons;
            }

            return hidden;
        }
    }
}
