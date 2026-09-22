using System;

namespace IrisNeuralNet.Diagnostics
{
    /// <summary>Метрики одной эпохи обучения.</summary>
    public readonly struct EpochMetrics
    {
        public EpochMetrics(int epoch, float trainLoss, float trainAccuracy, float validationLoss, float validationAccuracy)
        {
            Epoch = epoch;
            TrainLoss = trainLoss;
            TrainAccuracy = trainAccuracy;
            ValidationLoss = validationLoss;
            ValidationAccuracy = validationAccuracy;
        }

        public int Epoch { get; }
        public float TrainLoss { get; }
        public float TrainAccuracy { get; }
        public float ValidationLoss { get; }
        public float ValidationAccuracy { get; }
    }
}
