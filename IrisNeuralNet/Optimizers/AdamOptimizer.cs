using System;
using IrisNeuralNet.Core;

namespace IrisNeuralNet.Optimizers;

/// <summary>Adam: моменты m, v с коррекцией смещения. Значения по умолчанию — как в статье и Keras.</summary>
public sealed class AdamOptimizer : OptimizerBase
{
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _epsilon;
    private readonly float[][] _m;
    private readonly float[][] _v;
    private int _step;
    private float _biasCorrection1;
    private float _biasCorrection2;

    public AdamOptimizer(
        NeuralNetwork network,
        float learningRate = 0.001f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float epsilon = 1e-8f)
        : base(learningRate, network ?? throw new ArgumentNullException(nameof(network)))
    {
        if (beta1 < 0f || beta1 >= 1f) throw new ArgumentOutOfRangeException(nameof(beta1));
        if (beta2 < 0f || beta2 >= 1f) throw new ArgumentOutOfRangeException(nameof(beta2));
        if (!(epsilon > 0f)) throw new ArgumentOutOfRangeException(nameof(epsilon));

        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _m = OptimizerState.Create(network);
        _v = OptimizerState.Create(network);
    }

    protected override void BeginStep()
    {
        _step++;
        _biasCorrection1 = 1f - MathF.Pow(_beta1, _step);
        _biasCorrection2 = 1f - MathF.Pow(_beta2, _step);
    }

    protected override void Update(Span<float> parameters, ReadOnlySpan<float> gradients, int tensorIndex)
    {
        Span<float> m = _m[tensorIndex];
        Span<float> v = _v[tensorIndex];
        float learningRate = LearningRate;
        for (int i = 0; i < parameters.Length; i++)
        {
            float gradient = gradients[i];
            m[i] = _beta1 * m[i] + (1f - _beta1) * gradient;
            v[i] = _beta2 * v[i] + (1f - _beta2) * gradient * gradient;

            float mHat = m[i] / _biasCorrection1;
            float vHat = v[i] / _biasCorrection2;
            parameters[i] -= learningRate * mHat / (MathF.Sqrt(vHat) + _epsilon);
        }
    }
}
