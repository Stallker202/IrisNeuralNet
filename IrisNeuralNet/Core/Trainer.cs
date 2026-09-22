using IrisNeuralNet.Diagnostics;
using IrisNeuralNet.MathCore;
using IrisNeuralNet.Optimizers;
using System;

namespace IrisNeuralNet.Core
{
    /// <summary>
    /// Mini-batch тренер: перемешивает выборку, гоняет forward/backward/step.
    /// Все буферы grow-only: после прогрева цикл обучения не аллоцирует.
    /// </summary>
    public sealed class Trainer
    {
        private readonly NeuralNetwork _network;
        private readonly IOptimizer _optimizer;
        private readonly ILossFunction _loss;
        private readonly ITrainingObserver _observer;
        private float[] _batchX = Array.Empty<float>();
        private float[] _batchY = Array.Empty<float>();
        private float[] _predicted = Array.Empty<float>();
        private float[] _outputGradient = Array.Empty<float>();
        private int[] _order = Array.Empty<int>();

        public Trainer(NeuralNetwork network, IOptimizer optimizer, ILossFunction loss)
            : this(network, optimizer, loss, NullTrainingObserver.Instance)
        {
        }

        public Trainer(NeuralNetwork network, IOptimizer optimizer, ILossFunction loss, ITrainingObserver observer)
        {
            _network = network ?? throw new ArgumentNullException(nameof(network));
            _optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));
            _loss = loss ?? throw new ArgumentNullException(nameof(loss));
            _observer = observer ?? throw new ArgumentNullException(nameof(observer));
        }

        public float Fit(ReadOnlySpan<float> trainX, ReadOnlySpan<float> trainY, int epochs, int batchSize, int seed) =>
            Fit(trainX, trainY, ReadOnlySpan<float>.Empty, ReadOnlySpan<float>.Empty, epochs, batchSize, seed);

        public float Fit(
            ReadOnlySpan<float> trainX,
            ReadOnlySpan<float> trainY,
            ReadOnlySpan<float> validationX,
            ReadOnlySpan<float> validationY,
            int epochs,
            int batchSize,
            int seed)
        {
            EnsureDataset(trainX, trainY);
            bool hasValidation = !validationX.IsEmpty && !validationY.IsEmpty;
            if (hasValidation)
            {
                EnsureDataset(validationX, validationY);
            }

            if (epochs <= 0) throw new ArgumentOutOfRangeException(nameof(epochs));
            if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));

            int rowCount = trainY.Length / _network.OutputDim;
            int[] order = EnsureOrder(rowCount);
            var random = new Random(seed);

            EpochMetrics metrics = default;
            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                Shuffler.Shuffle(order.AsSpan(0, rowCount), random);
                for (int start = 0; start < rowCount; start += batchSize)
                {
                    int count = Math.Min(batchSize, rowCount - start);
                    TrainBatch(trainX, trainY, order, start, count);
                }

                metrics = EvaluateEpoch(epoch, trainX, trainY, validationX, validationY, hasValidation);
                _observer.OnEpochCompleted(in metrics);
            }

            return metrics.TrainLoss;
        }

        public float EvaluateLoss(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
        {
            EnsureDataset(x, y);
            Span<float> predicted = Predict(x);
            return _loss.Compute(predicted, y, _network.OutputDim);
        }

        public float EvaluateAccuracy(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
        {
            EnsureDataset(x, y);
            Span<float> predicted = Predict(x);
            return Metrics.Accuracy(predicted, y, _network.OutputDim);
        }

        private EpochMetrics EvaluateEpoch(
            int epoch,
            ReadOnlySpan<float> trainX,
            ReadOnlySpan<float> trainY,
            ReadOnlySpan<float> validationX,
            ReadOnlySpan<float> validationY,
            bool hasValidation)
        {
            Span<float> predicted = Predict(trainX);
            float trainLoss = _loss.Compute(predicted, trainY, _network.OutputDim);
            float trainAccuracy = Metrics.Accuracy(predicted, trainY, _network.OutputDim);

            float validationLoss = float.NaN;
            float validationAccuracy = float.NaN;
            if (hasValidation)
            {
                Span<float> validationPredicted = Predict(validationX);
                validationLoss = _loss.Compute(validationPredicted, validationY, _network.OutputDim);
                validationAccuracy = Metrics.Accuracy(validationPredicted, validationY, _network.OutputDim);
            }

            return new EpochMetrics(epoch, trainLoss, trainAccuracy, validationLoss, validationAccuracy);
        }

        private void TrainBatch(ReadOnlySpan<float> trainX, ReadOnlySpan<float> trainY, int[] order, int start, int count)
        {
            int inputWidth = _network.InputDim;
            int rowWidth = _network.OutputDim;

            Span<float> batchX = Buffers.Rent(ref _batchX, count * inputWidth);
            Span<float> batchY = Buffers.Rent(ref _batchY, count * rowWidth);
            for (int i = 0; i < count; i++)
            {
                int row = order[start + i];
                trainX.Slice(row * inputWidth, inputWidth).CopyTo(batchX.Slice(i * inputWidth));
                trainY.Slice(row * rowWidth, rowWidth).CopyTo(batchY.Slice(i * rowWidth));
            }

            Span<float> predicted = Buffers.Rent(ref _predicted, count * rowWidth);
            _network.Forward(batchX, predicted, count);

            Span<float> outputGradient = Buffers.Rent(ref _outputGradient, count * rowWidth);
            _loss.Backward(predicted, batchY, rowWidth, outputGradient);
            _network.Backward(outputGradient, count);
            _optimizer.Step(_network);
        }

        private Span<float> Predict(ReadOnlySpan<float> x)
        {
            int rowCount = x.Length / _network.InputDim;
            Span<float> predicted = Buffers.Rent(ref _predicted, rowCount * _network.OutputDim);
            _network.Forward(x, predicted, rowCount);
            return predicted;
        }

        private void EnsureDataset(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
        {
            SpanGuard.EnsureDivisible(x.Length, _network.InputDim, nameof(x));
            SpanGuard.EnsureDivisible(y.Length, _network.OutputDim, nameof(y));
            if (x.Length / _network.InputDim != y.Length / _network.OutputDim)
            {
                throw new ArgumentException("X and Y must have the same row count.");
            }
        }

        private int[] EnsureOrder(int rowCount)
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
}
