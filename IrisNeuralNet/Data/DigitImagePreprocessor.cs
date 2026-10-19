using System;

namespace IrisNeuralNet.Data;

/// <summary>
/// Приводит произвольное изображение цифры к виду MNIST, иначе сеть видит «чужое» распределение:
/// светлая цифра на тёмном фоне, ограничивающий прямоугольник вписан в 20×20 с сохранением
/// пропорций и помещён в поле 28×28 так, чтобы центр масс был в центре (14, 14).
/// Это та же процедура, которой готовили сам датасет MNIST.
/// </summary>
public static class DigitImagePreprocessor
{
    public const int Side = MnistDataLoader.ImageSide;

    private const int DigitBox = 20;
    private const float InkThresholdRatio = 0.1f;
    private const float MaxIntensity = 255f;

    /// <summary>
    /// Переводит оттенки серого (0 — чёрный, 255 — белый) в «чернила» MNIST:
    /// если фон (рамка изображения) светлый — типичная тёмная цифра на бумаге — инвертирует.
    /// </summary>
    public static void ToInk(Span<float> grayscale, int width, int height)
    {
        EnsureImage(grayscale, width, height);

        double borderSum = 0;
        int borderCount = 0;
        for (int x = 0; x < width; x++)
        {
            borderSum += grayscale[x] + grayscale[(height - 1) * width + x];
            borderCount += 2;
        }

        for (int y = 1; y < height - 1; y++)
        {
            borderSum += grayscale[y * width] + grayscale[y * width + width - 1];
            borderCount += 2;
        }

        if (borderSum / borderCount > MaxIntensity / 2)
        {
            for (int i = 0; i < grayscale.Length; i++) grayscale[i] = MaxIntensity - grayscale[i];
        }
    }

    /// <param name="ink">Интенсивности 0..255 (светлые штрихи на тёмном фоне), row-major width×height.</param>
    /// <returns>784 пикселя 0..255 в формате MNIST; нули, если изображение пустое.</returns>
    public static float[] ToMnist(ReadOnlySpan<float> ink, int width, int height)
    {
        EnsureImage(ink, width, height);

        var canvas = new float[Side * Side];
        float max = 0f;
        foreach (float value in ink) max = MathF.Max(max, value);
        if (max <= 0f) return canvas;

        if (!TryFindBoundingBox(ink, width, height, max * InkThresholdRatio, out int left, out int top, out int boxWidth, out int boxHeight))
        {
            return canvas;
        }

        float scale = DigitBox / (float)Math.Max(boxWidth, boxHeight);
        int patchWidth = Math.Clamp((int)MathF.Round(boxWidth * scale), 1, DigitBox);
        int patchHeight = Math.Clamp((int)MathF.Round(boxHeight * scale), 1, DigitBox);
        float[] patch = ResampleArea(ink, width, left, top, boxWidth, boxHeight, patchWidth, patchHeight);

        CenterOfMass(patch, patchWidth, patchHeight, out float centerX, out float centerY);
        int offsetX = Math.Clamp((int)MathF.Round(Side / 2f - centerX), 0, Side - patchWidth);
        int offsetY = Math.Clamp((int)MathF.Round(Side / 2f - centerY), 0, Side - patchHeight);

        float patchMax = 0f;
        foreach (float value in patch) patchMax = MathF.Max(patchMax, value);
        float gain = patchMax > 0f ? MaxIntensity / patchMax : 0f;

        for (int y = 0; y < patchHeight; y++)
        {
            for (int x = 0; x < patchWidth; x++)
            {
                canvas[(offsetY + y) * Side + offsetX + x] = patch[y * patchWidth + x] * gain;
            }
        }

        return canvas;
    }

    private static bool TryFindBoundingBox(
        ReadOnlySpan<float> ink, int width, int height, float threshold,
        out int left, out int top, out int boxWidth, out int boxHeight)
    {
        int minX = width, minY = height, maxX = -1, maxY = -1;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (ink[y * width + x] > threshold)
                {
                    minX = Math.Min(minX, x);
                    maxX = Math.Max(maxX, x);
                    minY = Math.Min(minY, y);
                    maxY = Math.Max(maxY, y);
                }
            }
        }

        left = minX;
        top = minY;
        boxWidth = maxX - minX + 1;
        boxHeight = maxY - minY + 1;
        return maxX >= 0;
    }

    /// <summary>
    /// Ресэмплинг усреднением по площади (box filter): каждый выходной пиксель — среднее
    /// исходных пикселей, взвешенных долей перекрытия. Корректен и для уменьшения, и для увеличения.
    /// </summary>
    private static float[] ResampleArea(
        ReadOnlySpan<float> source, int sourceWidth,
        int left, int top, int boxWidth, int boxHeight,
        int targetWidth, int targetHeight)
    {
        var target = new float[targetWidth * targetHeight];
        float stepX = boxWidth / (float)targetWidth;
        float stepY = boxHeight / (float)targetHeight;
        int right = left + boxWidth;
        int bottom = top + boxHeight;

        for (int ty = 0; ty < targetHeight; ty++)
        {
            float y0 = top + ty * stepY;
            float y1 = y0 + stepY;
            for (int tx = 0; tx < targetWidth; tx++)
            {
                float x0 = left + tx * stepX;
                float x1 = x0 + stepX;
                double sum = 0;
                double area = 0;
                for (int sy = (int)y0; sy < Math.Min((int)MathF.Ceiling(y1), bottom); sy++)
                {
                    float coverY = MathF.Min(y1, sy + 1) - MathF.Max(y0, sy);
                    for (int sx = (int)x0; sx < Math.Min((int)MathF.Ceiling(x1), right); sx++)
                    {
                        float coverage = coverY * (MathF.Min(x1, sx + 1) - MathF.Max(x0, sx));
                        sum += source[sy * sourceWidth + sx] * coverage;
                        area += coverage;
                    }
                }

                target[ty * targetWidth + tx] = area > 0 ? (float)(sum / area) : 0f;
            }
        }

        return target;
    }

    /// <summary>Центр масс интенсивности; координаты центров пикселей — x + 0.5.</summary>
    private static void CenterOfMass(float[] image, int width, int height, out float centerX, out float centerY)
    {
        double mass = 0, sumX = 0, sumY = 0;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float value = image[y * width + x];
                mass += value;
                sumX += value * (x + 0.5);
                sumY += value * (y + 0.5);
            }
        }

        centerX = mass > 0 ? (float)(sumX / mass) : width / 2f;
        centerY = mass > 0 ? (float)(sumY / mass) : height / 2f;
    }

    private static void EnsureImage(ReadOnlySpan<float> pixels, int width, int height)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if (pixels.Length != width * height)
        {
            throw new ArgumentException($"Expected {width * height} pixels for a {width}x{height} image.", nameof(pixels));
        }
    }
}
