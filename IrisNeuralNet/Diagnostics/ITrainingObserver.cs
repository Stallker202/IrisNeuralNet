using System;

namespace IrisNeuralNet.Diagnostics
{
    /// <summary>Получает метрики после каждой эпохи. Ядро не знает, кто их потребляет.</summary>
    public interface ITrainingObserver
    {
        void OnEpochCompleted(in EpochMetrics metrics);
    }
}
