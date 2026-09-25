using IrisNeuralNet.Core;
using Xunit;


namespace IrisNeuralNet.Tests
{
    public sealed class DenseLayerTests
    {
        [Fact]
        public void Forward_AppliesReLUElementwise()
        {
            var layer = LayerFactory.Identity(2, new ReLUActivation());
            float[] output = new float[2];

            layer.Forward(new float[] { -1f, 2f }, output, batchSize: 1);

            Assert.Equal(new float[] { 0f, 2f }, output);
        }

        [Fact]
        public void Forward_AddsBiasesBeforeActivation()
        {
            var layer = LayerFactory.Identity(2, new ReLUActivation(), biases: new float[] { 5f, -3f });
            float[] output = new float[2];

            layer.Forward(new float[] { 0f, 0f }, output, batchSize: 1);

            Assert.Equal(new float[] { 5f, 0f }, output);
        }

        [Fact]
        public void Forward_MultipliesWeightsRowMajor()
        {
            var layer = new DenseLayer(new float[] { 1f, 2f, 3f, 4f }, new float[] { 0f, 0f }, new ReLUActivation());
            float[] output = new float[2];

            layer.Forward(new float[] { 1f, 1f }, output, batchSize: 1);

            Assert.Equal(new float[] { 4f, 6f }, output);
        }

        [Fact]
        public void Forward_ProcessesBatchRowsIndependently()
        {
            var layer = LayerFactory.Identity(2, new ReLUActivation());
            float[] output = new float[4];

            layer.Forward(new float[] { -1f, 2f, 3f, -4f }, output, batchSize: 2);

            Assert.Equal(new float[] { 0f, 2f, 3f, 0f }, output);
        }

        [Fact]
        public void Forward_SoftmaxNormalizesEachRow()
        {
            var layer = LayerFactory.Identity(2, new SoftmaxActivation());
            float[] output = new float[2];

            layer.Forward(new float[] { 5f, 5f }, output, batchSize: 1);

            Assert.Equal(new float[] { 0.5f, 0.5f }, output);
        }

        [Fact]
        public void Constructor_Throws_WhenWeightsDoNotMatchBiases()
        {
            Assert.Throws<System.ArgumentException>(
                () => new DenseLayer(new float[3], new float[2], new ReLUActivation()));
        }

        [Fact]
        public void Backward_ComputesHandCheckedGradients()
        {
            var layer = new DenseLayer(new float[] { 1f, 1f }, new float[] { 0f }, new ReLUActivation());
            float[] output = new float[1];
            float[] inputGradient = new float[2];

            layer.Forward(new float[] { 1f, 2f }, output, batchSize: 1);
            layer.Backward(new float[] { 1f }, inputGradient, batchSize: 1);

            Assert.Equal(new float[] { 1f, 2f }, layer.WeightGradients.ToArray());
            Assert.Equal(new float[] { 1f }, layer.BiasGradients.ToArray());
            Assert.Equal(new float[] { 1f, 1f }, inputGradient);
        }

        [Fact]
        public void Backward_Throws_WhenBatchSizeDiffersFromForward()
        {
            var layer = LayerFactory.Identity(2, new ReLUActivation());
            float[] output = new float[2];
            layer.Forward(new float[] { 1f, 2f }, output, batchSize: 1);

            Assert.Throws<System.InvalidOperationException>(
                () => layer.Backward(new float[] { 1f, 1f }, new float[] { 0f, 0f, 0f, 0f }, batchSize: 2));
        }

        [Fact]
        public void Backward_WorksWithSmallerBatchAfterLargerForward()
        {
            var layer = new DenseLayer(new float[] { 1f }, new float[] { 0f }, new ReLUActivation());

            // Растим буферы батчем размера 2.
            layer.Forward(new float[] { 1f, 2f }, new float[2], batchSize: 2);

            // Рабочий батч размера 1.
            layer.Forward(new float[] { 3f }, new float[1], batchSize: 1);
            float[] inputGradient = new float[1];
            layer.Backward(new float[] { 1f }, inputGradient, batchSize: 1);

            // z = 3 > 0 → delta = 1; dW = x * delta = 3; dB = 1; dX = w * delta = 1.
            Assert.Equal(new float[] { 3f }, layer.WeightGradients.ToArray());
            Assert.Equal(new float[] { 1f }, layer.BiasGradients.ToArray());
            Assert.Equal(new float[] { 1f }, inputGradient);
        }

        [Fact]
        public void Probe_MatchesForwardForSingleSample()
        {
            var layer = new DenseLayer(new float[] { 1f, 2f, 3f, 4f }, new float[] { 0.5f, -0.5f }, new ReLUActivation());
            float[] forward = new float[2];
            float[] probe = new float[2];

            layer.Forward(new float[] { 1f, 2f }, forward, batchSize: 1);
            layer.Probe(new float[] { 1f, 2f }, probe);

            Assert.Equal(forward, probe);
        }
    }
}
