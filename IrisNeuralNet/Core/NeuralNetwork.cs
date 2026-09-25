using System;
using IrisNeuralNet.MathCore;

namespace IrisNeuralNet.Core;

/// <summary>
/// Последовательная сеть слоёв (аналог Sequential из Keras).
/// Прямой проход использует scratch активаций, обратный — отдельный scratch
/// градиентов: слой полностью читает свой dA до записи dX в тот же буфер,
/// поэтому алиасинг срезов безопасен.
/// </summary>
public sealed class NeuralNetwork
{
    private readonly DenseLayer[] _layers;
    private readonly int _intermediateRowWidth;
    private readonly int _gradientRowWidth;
    private float[] _scratch = Array.Empty<float>();
    private float[] _gradientScratch = Array.Empty<float>();

    public ReadOnlySpan<DenseLayer> Layers => _layers;

    public NeuralNetwork(DenseLayer[] layers)
    {
        if (layers is null) throw new ArgumentNullException(nameof(layers));
        if (layers.Length == 0) throw new ArgumentException("Network requires at least one layer.", nameof(layers));

        for (int i = 0; i < layers.Length - 1; i++)
        {
            if (layers[i].OutputDim != layers[i + 1].InputDim)
            {
                throw new ArgumentException(
                    $"Layer {i} output ({layers[i].OutputDim}) does not match layer {i + 1} input ({layers[i + 1].InputDim}).",
                    nameof(layers));
            }
        }

        int intermediateWidth = 0;
        int gradientWidth = 0;
        for (int i = 0; i < layers.Length; i++)
        {
            if (i < layers.Length - 1)
            {
                intermediateWidth = Math.Max(intermediateWidth, layers[i].OutputDim);
            }

            gradientWidth = Math.Max(gradientWidth, Math.Max(layers[i].InputDim, layers[i].OutputDim));
        }

        _layers = (DenseLayer[])layers.Clone();
        _intermediateRowWidth = intermediateWidth;
        _gradientRowWidth = gradientWidth;
    }

    public int InputDim => _layers[0].InputDim;
    public int OutputDim => _layers[_layers.Length - 1].OutputDim;
    public int LayerCount => _layers.Length;

    public void Forward(ReadOnlySpan<float> input, Span<float> output, int batchSize)
    {
        if (_layers.Length == 1)
        {
            _layers[0].Forward(input, output, batchSize);
            return;
        }

        Span<float> scratch = Buffers.Rent(ref _scratch, batchSize * _intermediateRowWidth);
        ReadOnlySpan<float> current = input;
        for (int i = 0; i < _layers.Length - 1; i++)
        {
            Span<float> target = scratch.Slice(0, batchSize * _layers[i].OutputDim);
            _layers[i].Forward(current, target, batchSize);
            current = target;
        }

        _layers[_layers.Length - 1].Forward(current, output, batchSize);
    }

    /// <summary>Обратный проход сверху вниз; градиенты накапливаются в слоях.</summary>
    public void Backward(ReadOnlySpan<float> outputGradient, int batchSize)
    {
        SpanGuard.EnsureLength(outputGradient, batchSize * OutputDim, nameof(outputGradient));

        Span<float> gradient = Buffers.Rent(ref _gradientScratch, batchSize * _gradientRowWidth);
        ReadOnlySpan<float> current = outputGradient;
        for (int i = _layers.Length - 1; i >= 0; i--)
        {
            Span<float> inputGradient = gradient.Slice(0, batchSize * _layers[i].InputDim);
            _layers[i].Backward(current, inputGradient, batchSize);
            current = inputGradient;
        }
    }

    private float[] _predictionBuffer = Array.Empty<float>();

    /// <summary>Прямой проход для одного образца: probabilities длиной OutputDim.</summary>
    public void Predict(ReadOnlySpan<float> features, Span<float> probabilities) =>
        Forward(features, probabilities, batchSize: 1);

    /// <summary>Индекс класса с максимальной вероятностью; ноль аллокаций в установившемся режиме.</summary>
    public int PredictClass(ReadOnlySpan<float> features)
    {
        Span<float> probabilities = Buffers.Rent(ref _predictionBuffer, OutputDim);
        Forward(features, probabilities, batchSize: 1);
        return Metrics.ArgMax(probabilities);
    }
}