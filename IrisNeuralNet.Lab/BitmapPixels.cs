using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using IrisNeuralNet.Lab.Scenarios;

namespace IrisNeuralNet.Lab;

/// <summary>Чтение и запись пикселей Bitmap как оттенков серого (0 — чёрный, 255 — белый).</summary>
internal static class BitmapPixels
{
    /// <summary>Яркость по формуле Rec. 601; прозрачность смешивается с белым фоном (как PNG «на бумаге»).</summary>
    public static float[] ToGrayscale(Bitmap bitmap)
    {
        if (bitmap is null) throw new ArgumentNullException(nameof(bitmap));

        int width = bitmap.Width;
        int height = bitmap.Height;
        var argb = new int[width * height];
        BitmapData data = bitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
        try
        {
            for (int y = 0; y < height; y++)
            {
                Marshal.Copy(data.Scan0 + y * data.Stride, argb, y * width, width);
            }
        }
        finally
        {
            bitmap.UnlockBits(data);
        }

        var gray = new float[argb.Length];
        for (int i = 0; i < argb.Length; i++)
        {
            int pixel = argb[i];
            float alpha = ((pixel >> 24) & 0xFF) / 255f;
            float luminance = 0.299f * ((pixel >> 16) & 0xFF) + 0.587f * ((pixel >> 8) & 0xFF) + 0.114f * (pixel & 0xFF);
            gray[i] = luminance * alpha + 255f * (1f - alpha);
        }

        return gray;
    }

    /// <summary>Изображение 28×28 из пикселей MNIST: inverted = чёрные чернила на белом (как plt.imshow, cmap=binary).</summary>
    public static Bitmap FromMnist(ReadOnlySpan<float> pixels, bool inverted)
    {
        int side = (int)Math.Sqrt(pixels.Length);
        var bitmap = new Bitmap(side, side, PixelFormat.Format32bppArgb);
        for (int y = 0; y < side; y++)
        {
            for (int x = 0; x < side; x++)
            {
                int value = (int)Math.Clamp(pixels[y * side + x], 0f, 255f);
                if (inverted) value = 255 - value;
                bitmap.SetPixel(x, y, Color.FromArgb(value, value, value));
            }
        }

        return bitmap;
    }
}

/// <summary>Загрузка png/jpg/bmp через System.Drawing для распознавания своих цифр (--image).</summary>
internal sealed class BitmapImageLoader : IImageLoader
{
    public float[] LoadGrayscale(string path, out int width, out int height)
    {
        using var bitmap = new Bitmap(path);
        width = bitmap.Width;
        height = bitmap.Height;
        return BitmapPixels.ToGrayscale(bitmap);
    }
}
