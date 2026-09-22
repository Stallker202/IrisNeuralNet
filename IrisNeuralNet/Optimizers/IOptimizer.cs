using IrisNeuralNet.Core;

namespace IrisNeuralNet.Optimizers
{
    /// <summary>Алгоритм обновления параметров по накопленным градиентам.</summary>
    public interface IOptimizer
    {
        void Step(NeuralNetwork network);
    }
}
