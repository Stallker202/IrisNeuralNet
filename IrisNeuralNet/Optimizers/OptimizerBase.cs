using System;
using IrisNeuralNet.Core;

namespace IrisNeuralNet.Optimizers;

/// <summary>
/// Общий каркас оптимизаторов (Template Method): валидация learning rate, привязка
/// stateful-оптимизатора к своей сети и обход тензоров параметров в фиксированном порядке
/// [W₀, b₀, W₁, b₁, …]. Наследник описывает только правило обновления одного тензора.
/// </summary>
public abstract class OptimizerBase : IOptimizer
{
    private readonly NeuralNetwork? _owner;
    private float _learningRate;

    /// <param name="owner">
    /// Сеть, под размеры которой выделено состояние (velocity, m, v). null — оптимизатор без состояния.
    /// </param>
    protected OptimizerBase(float learningRate, NeuralNetwork? owner)
    {
        LearningRate = learningRate;
        _owner = owner;
    }

    public float LearningRate
    {
        get => _learningRate;
        set
        {
            if (!(value > 0f) || float.IsInfinity(value))
            {
                throw new ArgumentOutOfRangeException(nameof(value), value, "Learning rate must be a positive finite number.");
            }

            _learningRate = value;
        }
    }

    public void Step(NeuralNetwork network)
    {
        if (network is null) throw new ArgumentNullException(nameof(network));
        if (_owner is not null && !ReferenceEquals(network, _owner))
        {
            throw new InvalidOperationException("Optimizer is bound to the network it was created for.");
        }

        BeginStep();
        int tensor = 0;
        foreach (DenseLayer layer in network.Layers)
        {
            Update(layer.Weights, layer.WeightGradients, tensor++);
            Update(layer.Biases, layer.BiasGradients, tensor++);
        }
    }

    /// <summary>Хук перед обновлением всех тензоров (например, счётчик шагов Adam).</summary>
    protected virtual void BeginStep()
    {
    }

    /// <param name="tensorIndex">Индекс тензора в порядке [W₀, b₀, W₁, b₁, …] — ключ к состоянию.</param>
    protected abstract void Update(Span<float> parameters, ReadOnlySpan<float> gradients, int tensorIndex);
}
