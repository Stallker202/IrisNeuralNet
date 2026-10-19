using System;
using IrisNeuralNet.Core;

namespace IrisNeuralNet.Optimizers;

/// <summary>Создание оптимизатора по идентификатору — единственное место сопоставления kind → класс.</summary>
public static class OptimizerFactory
{
    /// <summary>Инерция для SGD+Momentum: классическое значение 0.9.</summary>
    public const float DefaultMomentum = 0.9f;

    public static IOptimizer Create(OptimizerKind kind, NeuralNetwork network, float learningRate) =>
        kind switch
        {
            OptimizerKind.Adam => new AdamOptimizer(network, learningRate),
            OptimizerKind.RmsProp => new RmsPropOptimizer(network, learningRate),
            OptimizerKind.Sgd => new SgdOptimizer(learningRate),
            OptimizerKind.SgdMomentum => new SgdMomentumOptimizer(network, learningRate, DefaultMomentum),
            _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "Unknown optimizer."),
        };

    /// <summary>Learning rate по умолчанию, как в Keras (adam/rmsprop 0.001, sgd 0.01).</summary>
    public static float DefaultLearningRate(OptimizerKind kind) =>
        kind is OptimizerKind.Sgd or OptimizerKind.SgdMomentum ? 0.01f : 0.001f;

    /// <summary>Имя как в Keras: 'adam', 'rmsprop', 'sgd'.</summary>
    public static string KerasName(OptimizerKind kind) =>
        kind switch
        {
            OptimizerKind.Adam => "adam",
            OptimizerKind.RmsProp => "rmsprop",
            OptimizerKind.Sgd => "sgd",
            OptimizerKind.SgdMomentum => "sgd+momentum",
            _ => kind.ToString(),
        };
}
