using System;

namespace IrisNeuralNet.Core;

/// <summary>Единственное место, где сериализованный код превращается в полиморфный объект.</summary>
internal static class ActivationFactory
{
    public static ActivationFunction Create(ActivationKind kind) =>
        kind switch
        {
            ActivationKind.Relu => new ReLUActivation(),
            ActivationKind.Softmax => new SoftmaxActivation(),
            _ => throw new FormatException($"Unknown activation kind {(byte)kind}."),
        };
}
