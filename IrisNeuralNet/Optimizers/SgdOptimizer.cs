using IrisNeuralNet.Core;
using IrisNeuralNet.MathCore;
using System;


namespace IrisNeuralNet.Optimizers
{
    /// <summary>Классический градиентный спуск: p -= lr * g.</summary>
    public sealed class SgdOptimizer : IOptimizer
    {
        private float _learningRate;

        public float LearningRate
        {
            get => _learningRate;
            set
            {
                if (value <= 0f) throw new ArgumentOutOfRangeException(nameof(value));
                _learningRate = value;
            }
        }

        public SgdOptimizer(float learningRate)
        {
            if (learningRate <= 0f) throw new ArgumentOutOfRangeException(nameof(learningRate));
            _learningRate = learningRate;
        }

        public void Step(NeuralNetwork network)
        {
            foreach (DenseLayer layer in network.Layers)
            {
                VectorOps.MultiplyAdd(layer.WeightGradients, -_learningRate, layer.Weights);
                VectorOps.MultiplyAdd(layer.BiasGradients, -_learningRate, layer.Biases);
            }
        }
    }
}
