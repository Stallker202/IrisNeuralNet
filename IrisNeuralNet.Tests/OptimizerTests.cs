using IrisNeuralNet.Core;
using IrisNeuralNet.Optimizers;
using System;
using Xunit;

namespace IrisNeuralNet.Tests
{
    public sealed class OptimizerTests
    {
        [Fact]
        public void Sgd_MovesParametersAgainstGradient()
        {
            var (network, layer) = CreateNetworkWithGradients();
            float[] weightsBefore = layer.Weights.ToArray();
            float[] gradients = layer.WeightGradients.ToArray();

            new SgdOptimizer(0.1f).Step(network);

            for (int i = 0; i < weightsBefore.Length; i++)
            {
                Assert.Equal(weightsBefore[i] - 0.1f * gradients[i], layer.Weights[i], precision: 6);
            }
        }

        [Fact]
        public void Momentum_AccumulatesVelocityAcrossSteps()
        {
            var (network, layer) = CreateNetworkWithGradients();
            float[] weightsBefore = layer.Weights.ToArray();
            float[] gradients = layer.WeightGradients.ToArray();
            var optimizer = new SgdMomentumOptimizer(network, 0.1f, 0.9f);

            optimizer.Step(network);
            optimizer.Step(network);

            for (int i = 0; i < weightsBefore.Length; i++)
            {
                float v1 = -0.1f * gradients[i];
                float v2 = 0.9f * v1 - 0.1f * gradients[i];
                Assert.Equal(weightsBefore[i] + v1 + v2, layer.Weights[i], precision: 6);
            }
        }

        [Fact]
        public void Adam_FirstStepMovesByLearningRateTimesSign()
        {
            var (network, layer) = CreateNetworkWithGradients();
            float[] weightsBefore = layer.Weights.ToArray();
            float[] gradients = layer.WeightGradients.ToArray();

            new AdamOptimizer(network, learningRate: 0.001f).Step(network);

            for (int i = 0; i < weightsBefore.Length; i++)
            {
                Assert.Equal(weightsBefore[i] - 0.001f * MathF.Sign(gradients[i]), layer.Weights[i], precision: 6);
            }
        }

        [Fact]
        public void StatefulOptimizer_Throws_WhenSteppingForeignNetwork()
        {
            var (network, _) = CreateNetworkWithGradients();
            var (foreign, _) = CreateNetworkWithGradients();
            var optimizer = new SgdMomentumOptimizer(network, 0.1f, 0.9f);

            Assert.Throws<InvalidOperationException>(() => optimizer.Step(foreign));
        }

        private static (NeuralNetwork Network, DenseLayer Layer) CreateNetworkWithGradients()
        {
            var layer = LayerFactory.Identity(2, new ReLUActivation());
            var network = new NeuralNetwork(new[] { layer });
            float[] output = new float[2];

            layer.Forward(new float[] { 1f, 2f }, output, batchSize: 1);
            layer.Backward(new float[] { 1f, 1f }, new float[2], batchSize: 1);

            return (network, layer);
        }
    }
}
