using System;
using System.IO;
using IrisNeuralNet.Lab.Scenarios;

namespace IrisNeuralNet.Lab.Imaging;

/// <summary>Загрузка своей цифры для --image: PNG на любой ОС, без System.Drawing.</summary>
internal sealed class PngImageLoader : IImageLoader
{
    public float[] LoadGrayscale(string path, out int width, out int height)
    {
        try
        {
            return PngCodec.DecodeGrayscale(File.ReadAllBytes(path), out width, out height);
        }
        catch (FormatException e)
        {
            throw new FormatException($"'{path}': {e.Message} Сохраните изображение как PNG или откройте его во вкладке «Цифры» веб-интерфейса — браузер читает любые форматы.", e);
        }
    }
}
