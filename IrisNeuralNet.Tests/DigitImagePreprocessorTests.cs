using System;
using IrisNeuralNet.Data;
using Xunit;

namespace IrisNeuralNet.Tests;

public sealed class DigitImagePreprocessorTests
{
    [Fact]
    public void ToMnist_FitsDigitInto20BoxAndCentersMass()
    {
        // «Цифра» — прямоугольник 30×60 в углу картинки 100×100.
        const int side = 100;
        float[] ink = new float[side * side];
        for (int y = 5; y < 65; y++)
        {
            for (int x = 60; x < 90; x++) ink[y * side + x] = 255f;
        }

        float[] mnist = DigitImagePreprocessor.ToMnist(ink, side, side);

        Assert.Equal(784, mnist.Length);
        Measure(mnist, out int width, out int height, out double centerX, out double centerY, out float max);
        Assert.Equal(20, Math.Max(width, height));
        Assert.Equal(10, width);
        Assert.InRange(centerX, 13.0, 15.0);
        Assert.InRange(centerY, 13.0, 15.0);
        Assert.Equal(255f, max, precision: 3);
    }

    [Fact]
    public void ToMnist_ReturnsBlankForEmptyImage()
    {
        float[] mnist = DigitImagePreprocessor.ToMnist(new float[50 * 40], 50, 40);

        Assert.All(mnist, value => Assert.Equal(0f, value));
    }

    [Fact]
    public void ToInk_InvertsDarkDigitOnLightBackground()
    {
        float[] pixels = { 255f, 255f, 255f, 255f, 0f, 255f, 255f, 255f, 255f };

        DigitImagePreprocessor.ToInk(pixels, 3, 3);

        Assert.Equal(new float[] { 0f, 0f, 0f, 0f, 255f, 0f, 0f, 0f, 0f }, pixels);
    }

    [Fact]
    public void ToInk_KeepsLightDigitOnDarkBackground()
    {
        float[] pixels = { 0f, 0f, 0f, 0f, 200f, 0f, 0f, 0f, 0f };

        DigitImagePreprocessor.ToInk(pixels, 3, 3);

        Assert.Equal(200f, pixels[4]);
    }

    private static void Measure(float[] image, out int width, out int height, out double centerX, out double centerY, out float max)
    {
        int minX = 28, minY = 28, maxX = -1, maxY = -1;
        double mass = 0, sumX = 0, sumY = 0;
        max = 0f;
        for (int y = 0; y < 28; y++)
        {
            for (int x = 0; x < 28; x++)
            {
                float v = image[y * 28 + x];
                if (v <= 0f) continue;
                minX = Math.Min(minX, x); maxX = Math.Max(maxX, x);
                minY = Math.Min(minY, y); maxY = Math.Max(maxY, y);
                mass += v; sumX += v * (x + 0.5); sumY += v * (y + 0.5);
                max = MathF.Max(max, v);
            }
        }

        width = maxX - minX + 1;
        height = maxY - minY + 1;
        centerX = sumX / mass;
        centerY = sumY / mass;
    }
}
