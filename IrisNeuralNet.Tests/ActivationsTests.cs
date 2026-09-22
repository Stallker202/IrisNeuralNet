using IrisNeuralNet.MathCore;
using Xunit;

namespace IrisNeuralNet.Tests
{

    public sealed class ActivationsTests
    {
        [Fact]
        public void ReLU_ClampsNegativeValuesToZero()
        {
            float[] source = { -2f, -0.5f, 0f, 1.5f, 3f };
            float[] actual = new float[5];
    
            Activations.ReLU(source, actual);
    
            Assert.Equal(new float[] { 0f, 0f, 0f, 1.5f, 3f }, actual);
        }
    
        [Fact]
        public void Softmax_ReturnsProbabilityDistribution()
        {
            float[] actual = new float[3];
    
            Activations.Softmax(new float[] { 1f, 2f, 3f }, actual);
    
            Assert.Equal(1.0, VectorOps.Sum(actual), precision: 5);
            Assert.True(actual[0] < actual[1] && actual[1] < actual[2]);
        }
    
        [Fact]
        public void Softmax_IsInvariantToConstantShift()
        {
            float[] actual = new float[3];
            float[] shiftedActual = new float[3];
    
            Activations.Softmax(new float[] { 1f, 2f, 3f }, actual);
            Activations.Softmax(new float[] { 101f, 102f, 103f }, shiftedActual);
    
            for (int i = 0; i < actual.Length; i++)
            {
                Assert.Equal(actual[i], shiftedActual[i], precision: 4);
            }
        }
    
        [Fact]
        public void SoftmaxRows_NormalizesEachRowIndependently()
        {
            float[] actual = new float[4];
    
            Activations.SoftmaxRows(new float[] { 1f, 1f, 3f, 3f }, actual, columns: 2);
    
            Assert.Equal(new float[] { 0.5f, 0.5f, 0.5f, 0.5f }, actual);
        }
    
        [Fact]
        public void SoftmaxRows_MatchesHandComputedProbabilities()
        {
            float[] actual = new float[2];
    
            Activations.SoftmaxRows(new float[] { 1f, 3f }, actual, columns: 2);
    
            Assert.Equal(0.1192031f, actual[0], precision: 5);
            Assert.Equal(0.8807969f, actual[1], precision: 5);
        }

        [Fact]
        public void ReLUBackward_MasksNonPositiveSources()
        {
            float[] destination = new float[4];

            Activations.ReLUBackward(
                new float[] { -1f, 2f, 0f, 3f },
                new float[] { 5f, 6f, 7f, 8f },
                destination);

            Assert.Equal(new float[] { 0f, 6f, 0f, 8f }, destination);
        }

        [Fact]
        public void SoftmaxBackward_MatchesHandComputedJacobianProduct()
        {
            float[] destination = new float[2];

            Activations.SoftmaxBackward(
                new float[] { 1f, 3f },
                new float[] { 1f, 0f },
                destination);

            Assert.Equal(0.1049937f, destination[0], precision: 5);
            Assert.Equal(-0.1049937f, destination[1], precision: 5);
        }

        [Fact]
        public void SoftmaxBackward_GradientsSumToZero()
        {
            float[] destination = new float[2];

            Activations.SoftmaxBackward(
                new float[] { 1f, 3f },
                new float[] { 1f, 2f },
                destination);

            Assert.True(MathF.Abs(VectorOps.Sum(destination)) < 1e-6f);
        }
    }
}
