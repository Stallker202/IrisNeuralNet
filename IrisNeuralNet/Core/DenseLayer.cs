using System;
using IrisNeuralNet.MathCore;

namespace IrisNeuralNet.Core;

/// <summary>
/// Полносвязный слой: output = activation(input x weights + biases).
/// Веса row-major: [InputDim x OutputDim]. После Backward градиенты доступны
/// через WeightGradients / BiasGradients для оптимизатора.
/// Экземпляр не потокобезопасен: владеет рабочими буферами.
/// </summary>
public sealed class DenseLayer
{
    private readonly float[] _weights;
    private readonly float[] _biases;
    private readonly float[] _weightGradients;
    private readonly float[] _biasGradients;
    private readonly ActivationFunction _activation;
    private float[] _preActivation = Array.Empty<float>();
    private float[] _delta = Array.Empty<float>();
    private float[] _lastInput = Array.Empty<float>();
    private int _lastBatchSize;

    public int InputDim { get; }
    public int OutputDim { get; }
    public Span<float> Weights => _weights;
    public Span<float> Biases => _biases;
    public Span<float> WeightGradients => _weightGradients;
    public Span<float> BiasGradients => _biasGradients;

    public DenseLayer(float[] weights, float[] biases, ActivationFunction activation)
    {
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        if (biases is null) throw new ArgumentNullException(nameof(biases));
        if (biases.Length == 0) throw new ArgumentException("Biases must not be empty.", nameof(biases));
        if (weights.Length % biases.Length != 0)
        {
            throw new ArgumentException("Weights length must be a multiple of biases length.", nameof(weights));
        }

        _activation = activation ?? throw new ArgumentNullException(nameof(activation));
        _weights = (float[])weights.Clone();
        _biases = (float[])biases.Clone();
        _weightGradients = new float[weights.Length];
        _biasGradients = new float[biases.Length];
        OutputDim = biases.Length;
        InputDim = weights.Length / biases.Length;
    }

    /// <summary>Слой со случайной инициализацией Xavier (детерминированной по seed).</summary>
    public DenseLayer(int inputDim, int outputDim, ActivationFunction activation, int seed)
        : this(CreateXavierWeights(inputDim, outputDim, seed), new float[outputDim], activation)
    {
    }

    public void Forward(ReadOnlySpan<float> input, Span<float> output, int batchSize)
    {
        SpanGuard.EnsureLength(input, batchSize * InputDim, nameof(input));
        SpanGuard.EnsureLength(output, batchSize * OutputDim, nameof(output));

        input.CopyTo(Buffers.Rent(ref _lastInput, batchSize * InputDim));

        Span<float> preActivation = Buffers.Rent(ref _preActivation, batchSize * OutputDim);
        VectorOps.MatMul(input, _weights, preActivation, batchSize, InputDim, OutputDim);
        VectorOps.AddRowBroadcast(preActivation, _biases, OutputDim);
        _activation.Apply(preActivation, output, OutputDim);

        _lastBatchSize = batchSize;
    }

    /// <summary>
    /// Обратный проход: считает dW, dB (складывает в градиенты) и dX в inputGradient.
    /// Требует, чтобы Forward был вызван с тем же batchSize.
    /// </summary>
    public void Backward(ReadOnlySpan<float> outputGradient, Span<float> inputGradient, int batchSize)
    {
        if (batchSize != _lastBatchSize)
        {
            throw new InvalidOperationException("Forward must be called with the same batch size before Backward.");
        }

        SpanGuard.EnsureLength(outputGradient, batchSize * OutputDim, nameof(outputGradient));
        SpanGuard.EnsureLength(inputGradient, batchSize * InputDim, nameof(inputGradient));

        // Логический размер батча, а не ёмкость grow-only буфера!
        ReadOnlySpan<float> preActivation = _preActivation.AsSpan(0, batchSize * OutputDim);
        ReadOnlySpan<float> lastInput = _lastInput.AsSpan(0, batchSize * InputDim);

        Span<float> delta = Buffers.Rent(ref _delta, batchSize * OutputDim);
        _activation.Backward(preActivation, outputGradient, delta, OutputDim);

        VectorOps.MatMulWithTransposedLeft(lastInput, delta, _weightGradients, batchSize, InputDim, OutputDim);

        _biasGradients.AsSpan().Clear();
        for (int i = 0; i < batchSize; i++)
        {
            VectorOps.Add(delta.Slice(i * OutputDim, OutputDim), _biasGradients, _biasGradients);
        }

        for (int i = 0; i < batchSize; i++)
        {
            ReadOnlySpan<float> deltaRow = delta.Slice(i * OutputDim, OutputDim);
            Span<float> inputGradientRow = inputGradient.Slice(i * InputDim, InputDim);
            for (int k = 0; k < InputDim; k++)
            {
                inputGradientRow[k] = VectorOps.Dot(deltaRow, _weights.AsSpan(k * OutputDim, OutputDim));
            }
        }
    }

    private static float[] CreateXavierWeights(int inputDim, int outputDim, int seed)
    {
        if (inputDim <= 0) throw new ArgumentOutOfRangeException(nameof(inputDim));
        if (outputDim <= 0) throw new ArgumentOutOfRangeException(nameof(outputDim));

        var random = new Random(seed);
        float limit = MathF.Sqrt(6f / (inputDim + outputDim));
        var weights = new float[inputDim * outputDim];
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = (float)(random.NextDouble() * 2.0 - 1.0) * limit;
        }

        return weights;
    }
}