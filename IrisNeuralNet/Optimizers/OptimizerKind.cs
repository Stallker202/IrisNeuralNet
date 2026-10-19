namespace IrisNeuralNet.Optimizers;

/// <summary>Идентификатор оптимизатора для UI, конфигов экспериментов и отчётов (аналог optimizer='adam').</summary>
public enum OptimizerKind
{
    Adam,
    RmsProp,
    Sgd,
    SgdMomentum,
}
