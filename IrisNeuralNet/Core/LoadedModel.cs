using IrisNeuralNet.Data;

namespace IrisNeuralNet.Core;

/// <summary>Результат загрузки модели: сеть + опциональный нормализатор признаков.</summary>
public readonly record struct LoadedModel(NeuralNetwork Network, FeatureNormalizer? Normalizer);
