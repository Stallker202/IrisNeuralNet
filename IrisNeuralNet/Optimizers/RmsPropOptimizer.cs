using System;
using IrisNeuralNet.Core;

namespace IrisNeuralNet.Optimizers;

/// <summary>
/// RMSprop (как в Keras): v = ρ·v + (1 − ρ)·g²; p −= lr · g / (√v + ε).
/// Делит шаг на скользящее среднее квадрата градиента — адаптивная скорость по каждому параметру.
/// Значения по умолчанию совпадают с tf.keras.optimizers.RMSprop.
/// </summary>
public sealed class RmsPropOptimizer : OptimizerBase
{
    private readonly float _rho;
    private readonly float _epsilon;
    private readonly float[][] _meanSquare;

    public RmsPropOptimizer(NeuralNetwork network, float learningRate = 0.001f, float rho = 0.9f, float epsilon = 1e-7f)
        : base(learningRate, network ?? throw new ArgumentNullException(nameof(network)))
    {
        if (rho < 0f || rho >= 1f) throw new ArgumentOutOfRangeException(nameof(rho));
        if (!(epsilon > 0f)) throw new ArgumentOutOfRangeException(nameof(epsilon));

        _rho = rho;
        _epsilon = epsilon;
        _meanSquare = OptimizerState.Create(network);
    }

    protected override void Update(Span<float> parameters, ReadOnlySpan<float> gradients, int tensorIndex)
    {
        Span<float> meanSquare = _meanSquare[tensorIndex];
        float learningRate = LearningRate;
        for (int i = 0; i < parameters.Length; i++)
        {
            float gradient = gradients[i];
            meanSquare[i] = _rho * meanSquare[i] + (1f - _rho) * gradient * gradient;
            parameters[i] -= learningRate * gradient / (MathF.Sqrt(meanSquare[i]) + _epsilon);
        }
    }
}
