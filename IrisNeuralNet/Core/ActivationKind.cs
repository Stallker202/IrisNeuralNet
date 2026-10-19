namespace IrisNeuralNet.Core;

/// <summary>Сериализуемый код активации (байт в файле модели).</summary>
public enum ActivationKind : byte
{
    Relu = 1,
    Softmax = 2,
    Sigmoid = 3,
    Linear = 4,
}
