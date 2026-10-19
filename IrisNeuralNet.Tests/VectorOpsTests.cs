using System;
using IrisNeuralNet.MathCore;
using Xunit;

namespace IrisNeuralNet.Tests;

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

    [Fact]
    public void MatMulWithTransposedLeft_MatchesHandComputedProduct()
    {
        float[] left = { 1f, 2f, 3f, 4f, 5f, 6f };   // 3x2 → leftᵀ 2x3
        float[] right = { 1f, 0f, 0f, 1f, 1f, 1f };  // 3x2
        float[] actual = new float[4];

        VectorOps.MatMulWithTransposedLeft(left, right, actual, leftRows: 3, leftColumns: 2, rightColumns: 2);

        Assert.Equal(new float[] { 6f, 8f, 8f, 10f }, actual);
    }

    [Fact]
    public void MatMulWithTransposedRight_MatchesHandComputedProduct()
    {
        float[] left = { 1f, 2f, 3f, 4f };          // 2x2
        float[] right = { 5f, 6f, 7f, 8f, 9f, 10f }; // 3x2 → rightᵀ 2x3
        float[] actual = new float[6];

        VectorOps.MatMulWithTransposedRight(left, right, actual, leftRows: 2, inner: 2, rightRows: 3);

        Assert.Equal(new float[] { 17f, 23f, 29f, 39f, 53f, 67f }, actual);
    }

    [Fact]
    public void LargeProducts_UseParallelPathAndMatchNaiveReference()
    {
        // 160 × 200 × 160 ≈ 5.1M умножений > порога распараллеливания.
        const int rows = 160, inner = 200, columns = 160;
        var random = new Random(3);
        float[] left = Fill(random, rows * inner);
        float[] right = Fill(random, inner * columns);
        float[] product = new float[rows * columns];
        float[] gradientLeft = new float[inner * columns];
        float[] gradientRight = new float[rows * inner];

        VectorOps.MatMul(left, right, product, rows, inner, columns);
        VectorOps.MatMulWithTransposedLeft(left, product, gradientLeft, rows, inner, columns);
        VectorOps.MatMulWithTransposedRight(product, right, gradientRight, rows, columns, inner);

        for (int i = 0; i < rows; i += 17)
        {
            for (int j = 0; j < columns; j += 13)
            {
                double expected = 0;
                for (int k = 0; k < inner; k++) expected += left[i * inner + k] * right[k * columns + j];
                Assert.True(Math.Abs(expected - product[i * columns + j]) < 1e-3, $"product[{i},{j}]");
            }
        }

        for (int k = 0; k < inner; k += 19)
        {
            for (int j = 0; j < columns; j += 23)
            {
                double expected = 0;
                for (int i = 0; i < rows; i++) expected += left[i * inner + k] * product[i * columns + j];
                Assert.True(Math.Abs(expected - gradientLeft[k * columns + j]) < 1e-2 * (1 + Math.Abs(expected)), $"dW[{k},{j}]");
            }
        }

        for (int i = 0; i < rows; i += 29)
        {
            for (int k = 0; k < inner; k += 31)
            {
                double expected = 0;
                for (int j = 0; j < columns; j++) expected += product[i * columns + j] * right[k * columns + j];
                Assert.True(Math.Abs(expected - gradientRight[i * inner + k]) < 1e-2 * (1 + Math.Abs(expected)), $"dX[{i},{k}]");
            }
        }
    }

    [Fact]
    public void LargeMatMul_IsDeterministicAcrossCalls()
    {
        const int rows = 128, inner = 256, columns = 160;
        var random = new Random(4);
        float[] left = Fill(random, rows * inner);
        float[] right = Fill(random, inner * columns);
        float[] first = new float[rows * columns];
        float[] second = new float[rows * columns];

        VectorOps.MatMul(left, right, first, rows, inner, columns);
        VectorOps.MatMul(left, right, second, rows, inner, columns);

        Assert.Equal(first, second);
    }

    private static float[] Fill(Random random, int length)
    {
        float[] data = new float[length];
        for (int i = 0; i < length; i++) data[i] = (float)(random.NextDouble() * 2 - 1);
        return data;
    }
}
