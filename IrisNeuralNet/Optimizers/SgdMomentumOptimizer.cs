using IrisNeuralNet.Core;
using IrisNeuralNet.MathCore;
using System;


namespace IrisNeuralNet.Optimizers
{
    /// <summary>SGD с инерцией: v = momentum * v - lr * g; p += v.</summary>
    public sealed class SgdMomentumOptimizer : IOptimizer
    {
        private readonly NeuralNetwork _owner;
        private readonly float _learningRate;
        private readonly float _momentum;
        private readonly float[][] _velocityWeights;
        private readonly float[][] _velocityBiases;

        public SgdMomentumOptimizer(NeuralNetwork network, float learningRate, float momentum)
        {
            if (learningRate <= 0f) throw new ArgumentOutOfRangeException(nameof(learningRate));
            if (momentum < 0f || momentum >= 1f) throw new ArgumentOutOfRangeException(nameof(momentum));

            _owner = network ?? throw new ArgumentNullException(nameof(network));
            _learningRate = learningRate;
            _momentum = momentum;
            _velocityWeights = OptimizerState.ForWeights(network);
            _velocityBiases = OptimizerState.ForBiases(network);
        }

        public void Step(NeuralNetwork network)
        {
            if (!ReferenceEquals(network, _owner))
            {
                throw new InvalidOperationException("Optimizer is bound to the network it was created for.");
            }

            for (int i = 0; i < network.LayerCount; i++)
            {
                DenseLayer layer = network.Layers[i];

                Span<float> velocity = _velocityWeights[i];
                VectorOps.Scale(velocity, _momentum, velocity);
                VectorOps.MultiplyAdd(layer.WeightGradients, -_learningRate, velocity);
                VectorOps.Add(velocity, layer.Weights, layer.Weights);

                Span<float> biasVelocity = _velocityBiases[i];
                VectorOps.Scale(biasVelocity, _momentum, biasVelocity);
                VectorOps.MultiplyAdd(layer.BiasGradients, -_learningRate, biasVelocity);
                VectorOps.Add(biasVelocity, layer.Biases, layer.Biases);
            }
        }
    }
}
