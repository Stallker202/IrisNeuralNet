using System;
using System.Drawing;
using System.Globalization;
using IrisNeuralNet.Core;
using IrisNeuralNet.Data;
using IrisNeuralNet.Lab.Imaging;
using IrisNeuralNet.Lab.Rendering;
using IrisNeuralNet.Lab.Training;

namespace IrisNeuralNet.Lab.Web;

/// <summary>Откуда пришло изображение цифры: от этого зависит предобработка.</summary>
internal enum DigitSource
{
    /// <summary>Холст страницы: белые штрихи на чёрном — инверсия не нужна.</summary>
    Drawing,

    /// <summary>Файл пользователя: светлый фон инвертируется.</summary>
    Image,

    /// <summary>Уже готовые 28×28 в формате MNIST (тестовый образец, пересчёт во время обучения).</summary>
    Mnist,
}

/// <summary>
/// Пункт 10 методички MNIST — классификация пользовательского изображения: та же предобработка,
/// что у датасета (рамка 20×20, центр масс в центре 28×28), затем вероятности классов от сети,
/// которая сейчас обучается.
/// </summary>
internal static class DigitService
{
    public const int Side = MnistDataLoader.ImageSide;
    public const int ClassCount = MnistDataLoader.ClassCount;
    public const int MaxImageSide = 4096;

    /// <summary>Байты оттенков серого (0..255, построчно) → пиксели MNIST.</summary>
    public static float[] ToMnist(ReadOnlySpan<byte> gray, int width, int height, DigitSource source)
    {
        if (width is < 1 or > MaxImageSide || height is < 1 or > MaxImageSide || gray.Length != width * height)
        {
            throw new ArgumentException($"Ожидалось изображение до {MaxImageSide}×{MaxImageSide}: {width}×{height} пикселей.");
        }

        var pixels = new float[gray.Length];
        for (int i = 0; i < pixels.Length; i++) pixels[i] = gray[i];

        if (source == DigitSource.Mnist)
        {
            if (width != Side || height != Side) throw new ArgumentException($"Формат MNIST — {Side}×{Side}.");
            return pixels;
        }

        if (source == DigitSource.Image) DigitImagePreprocessor.ToInk(pixels, width, height);
        return DigitImagePreprocessor.ToMnist(pixels, width, height);
    }

    public static DigitResultDto Predict(LabData data, NeuralNetwork? network, float[] mnist, string? truth, float barsWidth, float barsHeight)
    {
        float[] probabilities = Array.Empty<float>();
        string verdict = "нет сети — нажми Start";
        if (network is not null && network.InputDim == mnist.Length)
        {
            float[] features = data.Prepared.Normalizer?.Transform(mnist, rowCount: 1) ?? mnist;

            // Своя проба: буферы сети делит поток обучения.
            probabilities = new NetworkProbe().Run(network, features).ToArray();
            int best = Metrics.ArgMax(probabilities);
            verdict = $"это {best}  ({probabilities[best].ToString("P1", CultureInfo.InvariantCulture)})";
        }


        var bars = new SvgBuilder(barsWidth, barsHeight);
        ProbabilityBarsPainter.Draw(bars, new RectangleF(0f, 0f, barsWidth, barsHeight), probabilities);

        byte[] bytes = DigitPng.ToBytes(mnist, inverted: false);
        return new DigitResultDto(
            Preview: PngCodec.ToDataUri(PngCodec.EncodeGray(bytes, Side, Side)),
            Pixels: Convert.ToBase64String(bytes),
            Probabilities: probabilities,
            Digit: probabilities.Length > 0 ? Metrics.ArgMax(probabilities) : -1,
            Verdict: verdict,
            Truth: truth ?? string.Empty,
            Bars: bars.ToString());
    }

    /// <summary>Случайный образец тестовой выборки: пиксели и подпись с истинной меткой.</summary>
    public static (float[] Pixels, string Truth) RandomTestSample(LabData data)
    {
        Dataset test = data.Prepared.RawValidation;
        int index = Random.Shared.Next(test.RowCount);
        return (test.Row(index).ToArray(), $"тест #{index}, истина: {test.Labels[index].ToString(CultureInfo.InvariantCulture)}");
    }
}
