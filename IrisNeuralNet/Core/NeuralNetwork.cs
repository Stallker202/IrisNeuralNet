using System;
using IrisNeuralNet.MathCore;

namespace IrisNeuralNet.Core;

/// <summary>
/// Последовательная сеть слоёв (аналог Sequential из Keras).
/// Обучающий Forward идёт через один scratch: слой копирует вход в свой кэш и считает z
/// в собственный буфер до записи выхода, поэтому алиасинг срезов безопасен.
/// Backward — отдельный scratch градиентов; первому слою dX не считается.
/// Infer — инференс без кэшей обучения через пару ping-pong буферов.
/// </summary>
public sealed class NeuralNetwork
{
    private readonly DenseLayer[] _layers;
    private readonly int _intermediateRowWidth;
    private readonly int _gradientRowWidth;
    private float[] _scratch = Array.Empty<float>();
    private float[] _gradientScratch = Array.Empty<float>();
    private float[] _inferencePing = Array.Empty<float>();
    private float[] _inferencePong = Array.Empty<float>();
    private float[] _predictionBuffer = Array.Empty<float>();

    public NeuralNetwork(DenseLayer[] layers)
    {
        if (layers is null) throw new ArgumentNullException(nameof(layers));
        if (layers.Length == 0) throw new ArgumentException("Network requires at least one layer.", nameof(layers));

        int intermediateWidth = 0;
        int gradientWidth = 0;
        for (int i = 0; i < layers.Length; i++)
        {
            if (layers[i] is null) throw new ArgumentException($"Layer {i} is null.", nameof(layers));

            if (i < layers.Length - 1)
            {
                if (layers[i].OutputDim != layers[i + 1].InputDim)
                {
                    throw new ArgumentException(
                        $"Layer {i} output ({layers[i].OutputDim}) does not match layer {i + 1} input ({layers[i + 1].InputDim}).",
                        nameof(layers));
                }

                intermediateWidth = Math.Max(intermediateWidth, layers[i].OutputDim);
            }

            if (i > 0)
            {
                // dX нужен только слоям выше первого.
                gradientWidth = Math.Max(gradientWidth, layers[i].InputDim);
            }
        }

        _layers = (DenseLayer[])layers.Clone();
        _intermediateRowWidth = intermediateWidth;
        _gradientRowWidth = gradientWidth;
    }

    public ReadOnlySpan<DenseLayer> Layers => _layers;
    public int InputDim => _layers[0].InputDim;
    public int OutputDim => _layers[_layers.Length - 1].OutputDim;
    public int LayerCount => _layers.Length;

    public int ParameterCount
    {
        get
        {
            int total = 0;
            foreach (DenseLayer layer in _layers) total += layer.ParameterCount;
            return total;
        }
    }

    /// <summary>Обучающий прямой проход: слои запоминают входы для Backward.</summary>
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
        for (int i = _layers.Length - 1; i > 0; i--)
        {
            Span<float> inputGradient = gradient.Slice(0, batchSize * _layers[i].InputDim);
            _layers[i].Backward(current, inputGradient, batchSize);
            current = inputGradient;
        }

        // Градиент по входу сети никому не нужен — первый слой его не считает.
        _layers[0].Backward(current, Span<float>.Empty, batchSize);
    }

    /// <summary>
    /// Инференс батча без побочных эффектов на кэши обучения.
    /// Использует собственные буферы сети, поэтому один экземпляр не вызывают из нескольких потоков.
    /// </summary>
    public void Infer(ReadOnlySpan<float> input, Span<float> output, int batchSize)
    {
        if (_layers.Length == 1)
        {
            _layers[0].Infer(input, output, batchSize);
            return;
        }

        Span<float> ping = Buffers.Rent(ref _inferencePing, batchSize * _intermediateRowWidth);
        Span<float> pong = Buffers.Rent(ref _inferencePong, batchSize * _intermediateRowWidth);
        ReadOnlySpan<float> current = input;
        for (int i = 0; i < _layers.Length - 1; i++)
        {
            Span<float> buffer = (i & 1) == 0 ? ping : pong;
            Span<float> target = buffer.Slice(0, batchSize * _layers[i].OutputDim);
            _layers[i].Infer(current, target, batchSize);
            current = target;
        }

        _layers[_layers.Length - 1].Infer(current, output, batchSize);
    }

    /// <summary>Выход сети для одного образца (вероятности классов или значение регрессии).</summary>
    public void Predict(ReadOnlySpan<float> features, Span<float> output) =>
        Infer(features, output, batchSize: 1);

    /// <summary>Индекс выхода с максимальным значением (класс softmax-сети); ноль аллокаций в установившемся режиме.</summary>
    public int PredictClass(ReadOnlySpan<float> features)
    {
        Span<float> probabilities = Buffers.Rent(ref _predictionBuffer, OutputDim);
        Infer(features, probabilities, batchSize: 1);
        return Metrics.ArgMax(probabilities);
    }
}
