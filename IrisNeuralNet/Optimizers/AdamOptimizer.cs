using IrisNeuralNet.Core;
using System;


namespace IrisNeuralNet.Optimizers
{
    public sealed class AdamOptimizer : IOptimizer
    {
        private readonly NeuralNetwork _owner;
        private readonly float _learningRate;
        private readonly float _beta1;
        private readonly float _beta2;
        private readonly float _epsilon;
        private readonly float[][] _mWeights;
        private readonly float[][] _vWeights;
        private readonly float[][] _mBiases;
        private readonly float[][] _vBiases;
        private int _step;

        public AdamOptimizer(
            NeuralNetwork network,
            float learningRate = 0.001f,
            float beta1 = 0.9f,
            float beta2 = 0.999f,
            float epsilon = 1e-8f)
        {
            if (learningRate <= 0f) throw new ArgumentOutOfRangeException(nameof(learningRate));

            _owner = network ?? throw new ArgumentNullException(nameof(network));
            _learningRate = learningRate;
            _beta1 = beta1;
            _beta2 = beta2;
            _epsilon = epsilon;
            _mWeights = OptimizerState.ForWeights(network);
            _vWeights = OptimizerState.ForWeights(network);
            _mBiases = OptimizerState.ForBiases(network);
            _vBiases = OptimizerState.ForBiases(network);
        }

        public void Step(NeuralNetwork network)
        {
            if (!ReferenceEquals(network, _owner))
            {
                throw new InvalidOperationException("Optimizer is bound to the network it was created for.");
            }

            _step++;
            float biasCorrection1 = 1f - MathF.Pow(_beta1, _step);
            float biasCorrection2 = 1f - MathF.Pow(_beta2, _step);

            for (int i = 0; i < network.LayerCount; i++)
            {
                DenseLayer layer = network.Layers[i];
                Update(layer.Weights, layer.WeightGradients, _mWeights[i], _vWeights[i], biasCorrection1, biasCorrection2);
                Update(layer.Biases, layer.BiasGradients, _mBiases[i], _vBiases[i], biasCorrection1, biasCorrection2);
            }
        }

        private void Update(
            Span<float> parameters,
            ReadOnlySpan<float> gradients,
            Span<float> m,
            Span<float> v,
            float biasCorrection1,
            float biasCorrection2)
        {
            for (int i = 0; i < parameters.Length; i++)
            {
                float gradient = gradients[i];
                m[i] = _beta1 * m[i] + (1f - _beta1) * gradient;
                v[i] = _beta2 * v[i] + (1f - _beta2) * gradient * gradient;

                float mHat = m[i] / biasCorrection1;
                float vHat = v[i] / biasCorrection2;
                parameters[i] -= _learningRate * mHat / (MathF.Sqrt(vHat) + _epsilon);
            }
        }
    }
}
