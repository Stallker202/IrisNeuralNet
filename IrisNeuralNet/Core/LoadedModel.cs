using IrisNeuralNet.Data;

namespace IrisNeuralNet.Core
{
    /// <summary>Результат загрузки: сеть + опциональный нормализатор признаков.</summary>
    public readonly struct LoadedModel
    {
        public LoadedModel(NeuralNetwork network, FeatureNormalizer? normalizer)
        {
            Network = network;
            Normalizer = normalizer;
        }

        public NeuralNetwork Network { get; }
        public FeatureNormalizer? Normalizer { get; }
    }
}
