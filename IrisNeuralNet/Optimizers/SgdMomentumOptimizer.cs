using System;
using IrisNeuralNet.Core;
using IrisNeuralNet.MathCore;

namespace IrisNeuralNet.Optimizers;

/// <summary>SGD с инерцией: v = μ·v − lr·g; p += v.</summary>
public sealed class SgdMomentumOptimizer : OptimizerBase
{
    private readonly float _momentum;
    private readonly float[][] _velocity;

    public SgdMomentumOptimizer(NeuralNetwork network, float learningRate, float momentum)
        : base(learningRate, network ?? throw new ArgumentNullException(nameof(network)))
    {
        if (momentum < 0f || momentum >= 1f) throw new ArgumentOutOfRangeException(nameof(momentum));

        _momentum = momentum;
        _velocity = OptimizerState.Create(network);
    }

    protected override void Update(Span<float> parameters, ReadOnlySpan<float> gradients, int tensorIndex)
    {
        Span<float> velocity = _velocity[tensorIndex];
        VectorOps.Scale(velocity, _momentum, velocity);
        VectorOps.MultiplyAdd(gradients, -LearningRate, velocity);
        VectorOps.Add(velocity, parameters, parameters);
    }
}
