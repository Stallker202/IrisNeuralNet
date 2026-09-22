using IrisNeuralNet.Core;

namespace IrisNeuralNet.Tests
{
    internal static class LayerFactory
    {
        public static DenseLayer Identity(int dim, ActivationFunction activation, float[]? biases = null)
        {
            var weights = new float[dim * dim];
            for (int i = 0; i < dim; i++)
            {
                weights[i * dim + i] = 1f;
            }

            return new DenseLayer(weights, biases ?? new float[dim], activation);
        }
    }
}
