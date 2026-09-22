using System;
using IrisNeuralNet.MathCore;
using Xunit;

namespace IrisNeuralNet.Tests
{
    public sealed class VectorOpsTests
    {
        [Fact]
        public void Add_SumsElementsPairwise()
        {
            float[] left = { 1f, 2f, 3f, 4f };
            float[] right = { 10f, 20f, 30f, 40f };
            float[] actual = new float[4];

            VectorOps.Add(left, right, actual);

            Assert.Equal(new float[] { 11f, 22f, 33f, 44f }, actual);
        }

        [Fact]
        public void Add_ProcessesTailBeyondVectorWidth()
        {
            const int length = 100;
            float[] left = new float[length];
            float[] right = new float[length];
            for (int i = 0; i < length; i++) { left[i] = i; right[i] = i; }
            float[] actual = new float[length];

            VectorOps.Add(left, right, actual);

            for (int i = 0; i < length; i++) Assert.Equal(2f * i, actual[i]);
        }

        [Fact]
        public void Add_Throws_WhenLengthsDiffer()
        {
            float[] left = { 1f, 2f };
            float[] right = { 1f };
            float[] actual = new float[2];

            Assert.Throws<ArgumentException>(() => VectorOps.Add(left, right, actual));
        }

        [Fact]
        public void Scale_MultipliesEveryElement()
        {
            float[] source = { 1f, 2f, 3f };
            float[] actual = new float[3];

            VectorOps.Scale(source, 0.5f, actual);

            Assert.Equal(new float[] { 0.5f, 1f, 1.5f }, actual);
        }

        [Fact]
        public void MultiplyAdd_AccumulatesScaledSourceIntoDestination()
        {
            float[] source = { 1f, 2f, 3f };
            float[] destination = { 10f, 20f, 30f };

            VectorOps.MultiplyAdd(source, 2f, destination);

            Assert.Equal(new float[] { 12f, 24f, 36f }, destination);
        }

        [Fact]
        public void Max_ReturnsLargestElement()
        {
            Assert.Equal(9.5f, VectorOps.Max(new float[] { -5f, 3f, 9.5f, 2f }));
        }

        [Fact]
        public void Sum_AddsAllElements()
        {
            Assert.Equal(10f, VectorOps.Sum(new float[] { 1f, 2f, 3f, 4f }));
        }

        [Fact]
        public void MatMul_MultipliesRowMajorMatrices()
        {
            float[] left = { 1f, 2f, 3f, 4f };    // 2x2
            float[] right = { 5f, 6f, 7f, 8f };  // 2x2
            float[] actual = new float[4];

            VectorOps.MatMul(left, right, actual, leftRows: 2, inner: 2, rightColumns: 2);

            Assert.Equal(new float[] { 19f, 22f, 43f, 50f }, actual);
        }

        [Fact]
        public void Dot_ComputesScalarProduct()
        {
            Assert.Equal(32f, VectorOps.Dot(new float[] { 1f, 2f, 3f }, new float[] { 4f, 5f, 6f }));
        }

        [Fact]
        public void Dot_ProcessesTailBeyondVectorWidth()
        {
            float[] left = { 1f, 2f, 3f, 4f, 5f };
            float[] right = { 2f, 2f, 2f, 2f, 2f };

            Assert.Equal(30f, VectorOps.Dot(left, right));
        }
    }
}
