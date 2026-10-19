using System;
using BenchmarkDotNet.Attributes;
using IrisNeuralNet.Core;

namespace IrisNeuralNet.Benchmarks;

/// <summary>
/// Слой MNIST-размера (батч 128, 784 → 512): здесь срабатывает распараллеливание матричных
/// произведений. Сравнение forward и полного forward+backward показывает цену обратного прохода
/// и выигрыш от пропуска dX у первого слоя.
/// </summary>
[MemoryDiagnoser]
public class DenseLayerBenchmarks
{
    private const int BatchSize = 128;
    private const int InputDim = 784;
    private const int OutputDim = 512;

    private DenseLayer _layer = null!;
    private float[] _input = Array.Empty<float>();
    private float[] _output = Array.Empty<float>();
    private float[] _outputGradient = Array.Empty<float>();
    private float[] _inputGradient = Array.Empty<float>();

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(1);
        _layer = new DenseLayer(InputDim, OutputDim, new ReLUActivation(), seed: 2);
        _input = new float[BatchSize * InputDim];
        _outputGradient = new float[BatchSize * OutputDim];
        for (int i = 0; i < _input.Length; i++) _input[i] = (float)random.NextDouble();
        for (int i = 0; i < _outputGradient.Length; i++) _outputGradient[i] = (float)(random.NextDouble() - 0.5);
        _output = new float[BatchSize * OutputDim];
        _inputGradient = new float[BatchSize * InputDim];
    }

    [Benchmark(Baseline = true)]
    public float Forward()
    {
        _layer.Forward(_input, _output, BatchSize);
        return _output[0];
    }

    [Benchmark]
    public float ForwardBackwardWithInputGradient()
    {
        _layer.Forward(_input, _output, BatchSize);
        _layer.Backward(_outputGradient, _inputGradient, BatchSize);
        return _layer.WeightGradients[0];
    }

    [Benchmark]
    public float ForwardBackwardFirstLayer()
    {
        _layer.Forward(_input, _output, BatchSize);
        _layer.Backward(_outputGradient, Span<float>.Empty, BatchSize);
        return _layer.WeightGradients[0];
    }
}
