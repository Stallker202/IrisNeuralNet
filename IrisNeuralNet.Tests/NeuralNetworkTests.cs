using IrisNeuralNet.Core;
using Xunit;

namespace IrisNeuralNet.Tests
{
    public sealed class NeuralNetworkTests
    {
        [Fact]
        public void Forward_ChainsLayersInOrder()
        {
            var network = new NeuralNetwork(new[]
            {
            LayerFactory.Identity(2, new ReLUActivation()),
            LayerFactory.Identity(2, new SoftmaxActivation()),
        });
            float[] output = new float[2];

            network.Forward(new float[] { -1f, 2f }, output, batchSize: 1);

            // relu([-1, 2]) = [0, 2]; softmax([0, 2]) = [1/(1+e^2), e^2/(1+e^2)]
            Assert.Equal(0.1192029f, output[0], precision: 5);
            Assert.Equal(0.8807971f, output[1], precision: 5);
        }

        [Fact]
        public void Forward_BatchThroughTwoLayers()
        {
            var network = new NeuralNetwork(new[]
            {
            LayerFactory.Identity(2, new ReLUActivation()),
            LayerFactory.Identity(2, new SoftmaxActivation()),
        });
            float[] output = new float[4];

            network.Forward(new float[] { -1f, 2f, 3f, -4f }, output, batchSize: 2);

            Assert.Equal(0.1192029f, output[0], precision: 5);
            Assert.Equal(0.8807971f, output[1], precision: 5);
            Assert.Equal(0.9525741f, output[2], precision: 5);
            Assert.Equal(0.0474259f, output[3], precision: 5);
        }

        [Fact]
        public void Forward_SingleLayerPassesThrough()
        {
            var network = new NeuralNetwork(new[] { LayerFactory.Identity(2, new ReLUActivation()) });
            float[] output = new float[2];

            network.Forward(new float[] { -1f, 2f }, output, batchSize: 1);

            Assert.Equal(new float[] { 0f, 2f }, output);
        }

        [Fact]
        public void Constructor_Throws_WhenLayerDimensionsMismatch()
        {
            Assert.Throws<System.ArgumentException>(
                () => new NeuralNetwork(new[]
                {
                LayerFactory.Identity(2, new ReLUActivation()),
                LayerFactory.Identity(3, new ReLUActivation()),
                }));
        }
    }
}
