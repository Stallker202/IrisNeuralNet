using System.Collections.Generic;

namespace IrisNeuralNet.Core
{



    public static class NetworkFactory
    {
        public static NeuralNetwork Create(int inputDim, int[] hiddenNeurons, int outputDim, ActivationFunction outputActivation, int seed)
        {
            var layers = new List<DenseLayer>();
            int input = inputDim;
            for (int i = 0; i < hiddenNeurons.Length; i++)
            {
                layers.Add(new DenseLayer(input, hiddenNeurons[i], new ReLUActivation(), seed: seed + i + 1));
                input = hiddenNeurons[i];
            }

            layers.Add(new DenseLayer(input, outputDim, outputActivation, seed: seed + hiddenNeurons.Length + 1));
            return new NeuralNetwork(layers.ToArray());
        }

        public static NeuralNetwork Create(int inputDim, int[] hiddenNeurons, int classCount, int seed) =>
            Create(inputDim, hiddenNeurons, classCount, new SoftmaxActivation(), seed);
    }
}