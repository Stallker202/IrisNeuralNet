using IrisNeuralNet.Core;

namespace IrisNeuralNet.Optimizers;

/// <summary>Алгоритм обновления параметров по накопленным градиентам.</summary>
public interface IOptimizer
{
    /// <summary>Скорость обучения; меняется на лету (live-слайдер UI), валидируется (&gt; 0).</summary>
    float LearningRate { get; set; }

    void Step(NeuralNetwork network);
}
