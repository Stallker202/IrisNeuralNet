using IrisNeuralNet.Core;

namespace IrisNeuralNet.Optimizers;

/// <summary>Выделяет по одному нулевому буферу на каждый тензор параметров сети [W₀, b₀, W₁, b₁, …].</summary>
internal static class OptimizerState
{
    public static float[][] Create(NeuralNetwork network)
    {
        var slots = new float[network.LayerCount * 2][];
        for (int i = 0; i < network.LayerCount; i++)
        {
            DenseLayer layer = network.Layers[i];
            slots[i * 2] = new float[layer.Weights.Length];
            slots[i * 2 + 1] = new float[layer.Biases.Length];
        }

        return slots;
    }
}
