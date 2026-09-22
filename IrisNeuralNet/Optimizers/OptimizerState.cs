using IrisNeuralNet.Core;
using System;


namespace IrisNeuralNet.Optimizers
{
    internal static class OptimizerState
    {
        public static float[][] ForWeights(NeuralNetwork network) => For(network, layer => layer.Weights.Length);

        public static float[][] ForBiases(NeuralNetwork network) => For(network, layer => layer.Biases.Length);

        private static float[][] For(NeuralNetwork network, Func<DenseLayer, int> size)
        {
            var slots = new float[network.LayerCount][];
            for (int i = 0; i < slots.Length; i++)
            {
                slots[i] = new float[size(network.Layers[i])];
            }

            return slots;
        }
    }
}
