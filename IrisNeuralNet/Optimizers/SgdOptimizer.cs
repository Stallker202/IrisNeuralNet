using System;
using IrisNeuralNet.MathCore;

namespace IrisNeuralNet.Optimizers;

/// <summary>Классический градиентный спуск: p −= lr · g. Состояния нет — подходит любой сети.</summary>
public sealed class SgdOptimizer : OptimizerBase
{
    public SgdOptimizer(float learningRate)
        : base(learningRate, owner: null)
    {
    }

    protected override void Update(Span<float> parameters, ReadOnlySpan<float> gradients, int tensorIndex) =>
        VectorOps.MultiplyAdd(gradients, -LearningRate, parameters);
}
