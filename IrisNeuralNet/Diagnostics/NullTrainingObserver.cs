using System;

namespace IrisNeuralNet.Diagnostics
{
    /// <summary>Special Case: наблюдатель «ничего не делать». Trainer не проверяет null.</summary>
    public sealed class NullTrainingObserver : ITrainingObserver
    {
        public static readonly NullTrainingObserver Instance = new();

        private NullTrainingObserver()
        {
        }

        public void OnEpochCompleted(in EpochMetrics metrics)
        {
        }
    }
}
