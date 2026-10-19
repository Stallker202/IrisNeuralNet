using System;
using System.Collections.Generic;
using IrisNeuralNet.Core;

namespace IrisNeuralNet.Lab.Training;

/// <summary>
/// Read-only проход одного образца для визуализации: собственные буферы, только DenseLayer.Infer,
/// поэтому UI-поток не трогает кэши обучающего потока.
/// </summary>
internal sealed class NetworkProbe
{
    private readonly List<float[]> _activations = new();

    /// <summary>Активации выходов слоёв последнего Run (индекс = номер слоя).</summary>
    public IReadOnlyList<float[]> Activations => _activations;

    public ReadOnlySpan<float> Run(NeuralNetwork network, ReadOnlySpan<float> input)
    {
        while (_activations.Count < network.LayerCount)
        {
            _activations.Add(Array.Empty<float>());
        }

        ReadOnlySpan<float> current = input;
        for (int i = 0; i < network.LayerCount; i++)
        {
            DenseLayer layer = network.Layers[i];
            float[] buffer = _activations[i];
            if (buffer.Length < layer.OutputDim)
            {
                buffer = new float[layer.OutputDim];
                _activations[i] = buffer;
            }

            layer.Infer(current, buffer.AsSpan(0, layer.OutputDim), batchSize: 1);
            current = buffer.AsSpan(0, layer.OutputDim);
        }

        return current;
    }
}
