using System;
using IrisNeuralNet.MathCore;

namespace IrisNeuralNet.Core;

/// <summary>
/// Полносвязный слой (Keras Dense): output = activation(input · weights + biases).
/// Веса row-major: [InputDim x OutputDim]. После Backward градиенты доступны
/// через WeightGradients / BiasGradients для оптимизатора.
/// Forward/Backward владеют рабочими буферами и не потокобезопасны;
/// <see cref="Infer"/> не трогает буферы обучения (только читает веса).
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

    public DenseLayer(float[] weights, float[] biases, ActivationFunction activation)
    {
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        if (biases is null) throw new ArgumentNullException(nameof(biases));
        if (biases.Length == 0) throw new ArgumentException("Biases must not be empty.", nameof(biases));
        if (weights.Length == 0 || weights.Length % biases.Length != 0)
        {
            throw new ArgumentException("Weights length must be a positive multiple of biases length.", nameof(weights));
        }

        _activation = activation ?? throw new ArgumentNullException(nameof(activation));
        _weights = (float[])weights.Clone();
        _biases = (float[])biases.Clone();
        _weightGradients = new float[weights.Length];
        _biasGradients = new float[biases.Length];
        OutputDim = biases.Length;
        InputDim = weights.Length / biases.Length;
    }

    /// <summary>Слой со случайной инициализацией Glorot/Xavier uniform (по умолчанию в Keras), детерминированной по seed.</summary>
    public DenseLayer(int inputDim, int outputDim, ActivationFunction activation, int seed)
        : this(CreateXavierWeights(inputDim, outputDim, seed), new float[outputDim], activation)
    {
    }

    public int InputDim { get; }
    public int OutputDim { get; }
    public int ParameterCount => _weights.Length + _biases.Length;
    public ActivationKind ActivationKind => _activation.Kind;
    public Span<float> Weights => _weights;
    public Span<float> Biases => _biases;
    public Span<float> WeightGradients => _weightGradients;
    public Span<float> BiasGradients => _biasGradients;

    /// <summary>Обучающий прямой проход: запоминает вход и пре-активацию для Backward.</summary>
    public void Forward(ReadOnlySpan<float> input, Span<float> output, int batchSize)
    {
        SpanGuard.EnsureLength(input, batchSize * InputDim, nameof(input));
        SpanGuard.EnsureLength(output, batchSize * OutputDim, nameof(output));

        input.CopyTo(Buffers.Rent(ref _lastInput, batchSize * InputDim));

        Span<float> preActivation = Buffers.Rent(ref _preActivation, batchSize * OutputDim);
        ComputePreActivation(input, preActivation, batchSize);
        _activation.Apply(preActivation, output, OutputDim);

        _lastBatchSize = batchSize;
    }

    /// <summary>
    /// Обратный проход: dW, dB — в градиенты слоя, dX — в inputGradient.
    /// Пустой inputGradient означает «dX не нужен» (первый слой сети): экономит матричное произведение.
    /// Требует, чтобы Forward был вызван с тем же batchSize.
    /// </summary>
    public void Backward(ReadOnlySpan<float> outputGradient, Span<float> inputGradient, int batchSize)
    {
        if (batchSize != _lastBatchSize)
        {
            throw new InvalidOperationException("Forward must be called with the same batch size before Backward.");
        }

        SpanGuard.EnsureLength(outputGradient, batchSize * OutputDim, nameof(outputGradient));
        bool needsInputGradient = !inputGradient.IsEmpty;
        if (needsInputGradient)
        {
            SpanGuard.EnsureLength(inputGradient, batchSize * InputDim, nameof(inputGradient));
        }

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

        if (needsInputGradient)
        {
            VectorOps.MatMulWithTransposedRight(delta, _weights, inputGradient, batchSize, OutputDim, InputDim);
        }
    }

    /// <summary>
    /// Прямой проход без записи в буферы обучения: безопасен для наблюдателей (UI)
    /// и для оценки, пока поток обучения не обновляет веса. input и output не должны пересекаться.
    /// </summary>
    public void Infer(ReadOnlySpan<float> input, Span<float> output, int batchSize)
    {
        SpanGuard.EnsureLength(input, batchSize * InputDim, nameof(input));
        SpanGuard.EnsureLength(output, batchSize * OutputDim, nameof(output));

        ComputePreActivation(input, output, batchSize);
        _activation.Apply(output, output, OutputDim);
    }

    private void ComputePreActivation(ReadOnlySpan<float> input, Span<float> destination, int batchSize)
    {
        VectorOps.MatMul(input, _weights, destination, batchSize, InputDim, OutputDim);
        VectorOps.AddRowBroadcast(destination, _biases, OutputDim);
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
