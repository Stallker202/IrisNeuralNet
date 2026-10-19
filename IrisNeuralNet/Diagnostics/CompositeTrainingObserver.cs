using System;

namespace IrisNeuralNet.Diagnostics;

/// <summary>Рассылает метрики эпохи нескольким наблюдателям (например, история + консоль).</summary>
public sealed class CompositeTrainingObserver : ITrainingObserver
{
    private readonly ITrainingObserver[] _observers;

    public CompositeTrainingObserver(params ITrainingObserver[] observers)
    {
        if (observers is null) throw new ArgumentNullException(nameof(observers));
        foreach (ITrainingObserver observer in observers)
        {
            if (observer is null) throw new ArgumentException("Observer must not be null.", nameof(observers));
        }

        _observers = (ITrainingObserver[])observers.Clone();
    }

    public void OnEpochCompleted(in EpochMetrics metrics)
    {
        foreach (ITrainingObserver observer in _observers)
        {
            observer.OnEpochCompleted(in metrics);
        }
    }
}
