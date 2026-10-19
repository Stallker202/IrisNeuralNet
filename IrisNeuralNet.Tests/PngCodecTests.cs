using System;
using System.Buffers.Binary;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using IrisNeuralNet.Lab.Imaging;
using Xunit;

namespace IrisNeuralNet.Tests;

public sealed class PngCodecTests
{
    [Fact]
    public void GrayRoundTrip_PreservesPixels()
    {
        byte[] pixels = Enumerable.Range(0, 28 * 28).Select(i => (byte)(i * 7)).ToArray();

        float[] decoded = PngCodec.DecodeGrayscale(PngCodec.EncodeGray(pixels, 28, 28), out int width, out int height);

        Assert.Equal((28, 28), (width, height));
        Assert.Equal(pixels.Select(p => (float)p), decoded);
    }

    [Fact]
    public void RgbaRoundTrip_UsesLuminanceAndWhiteBackground()
    {
        byte[] pixels =
        {
            255, 0, 0, 255,     // красный: 0.299 · 255
            0, 0, 0, 0,         // прозрачный → белый фон
            0, 0, 0, 128,       // полупрозрачный чёрный → серый
        };

        float[] decoded = PngCodec.DecodeGrayscale(PngCodec.EncodeRgba(pixels, 3, 1), out _, out _);

        Assert.Equal(76.245f, decoded[0], precision: 1);
        Assert.Equal(255f, decoded[1], precision: 3);
        Assert.Equal(127f, decoded[2], precision: 0);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void Decoder_UndoesEveryFilterType(byte filter)
    {
        // 3×2 RGB: строки отфильтрованы вручную тем же предсказателем, что и в стандарте PNG.
        byte[][] rows =
        {
            new byte[] { 10, 20, 30, 40, 50, 60, 70, 80, 90 },
            new byte[] { 15, 25, 35, 45, 55, 65, 75, 85, 95 },
        };
        byte[] raw = new byte[2 * 10];
        for (int y = 0; y < 2; y++)
        {
            raw[y * 10] = filter;
            for (int i = 0; i < 9; i++)
            {
                int left = i >= 3 ? rows[y][i - 3] : 0;
                int up = y > 0 ? rows[y - 1][i] : 0;
                int upLeft = y > 0 && i >= 3 ? rows[y - 1][i - 3] : 0;
                int predictor = filter switch
                {
                    1 => left,
                    2 => up,
                    3 => (left + up) / 2,
                    _ => Paeth(left, up, upLeft),
                };
                raw[y * 10 + 1 + i] = (byte)(rows[y][i] - predictor);
            }
        }

        float[] decoded = PngCodec.DecodeGrayscale(BuildPng(3, 2, bitDepth: 8, colorType: 2, raw), out _, out _);

        Assert.Equal(Luminance(10, 20, 30), decoded[0], precision: 3);
        Assert.Equal(Luminance(75, 85, 95), decoded[5], precision: 3);
    }

    [Fact]
    public void Decoder_ReadsOneBitPaletteWithTransparency()
    {
        // 8 пикселей в одном байте: 1,0,1,0,…; палитра: 0 — чёрный непрозрачный, 1 — красный прозрачный.
        byte[] raw = { 0, 0b1010_1010 };
        byte[] palette = { 0, 0, 0, 255, 0, 0 };
        byte[] transparency = { 255, 0 };

        float[] decoded = PngCodec.DecodeGrayscale(
            BuildPng(8, 1, bitDepth: 1, colorType: 3, raw, ("PLTE", palette), ("tRNS", transparency)), out int width, out _);

        Assert.Equal(8, width);
        Assert.Equal(new[] { 255f, 0f, 255f, 0f, 255f, 0f, 255f, 0f }, decoded);
    }

    [Fact]
    public void Decoder_TakesHighByteOf16BitSamples()
    {
        byte[] raw = { 0, 0xAB, 0xCD, 0x12, 0x34 };

        float[] decoded = PngCodec.DecodeGrayscale(BuildPng(2, 1, bitDepth: 16, colorType: 0, raw), out _, out _);

        Assert.Equal(new[] { 171f, 18f }, decoded);
    }

    [Fact]
    public void Decoder_RejectsBrokenFiles()
    {
        byte[] valid = PngCodec.EncodeGray(new byte[4], 2, 2);

        Assert.Throws<FormatException>(() => PngCodec.DecodeGrayscale(Encoding.ASCII.GetBytes("GIF89a......"), out _, out _));
        Assert.Throws<FormatException>(() => PngCodec.DecodeGrayscale(valid.AsSpan(0, valid.Length - 20), out _, out _));

        byte[] corrupted = (byte[])valid.Clone();
        corrupted[20] ^= 0xFF; // байт внутри IHDR — CRC не сойдётся
        Assert.Throws<FormatException>(() => PngCodec.DecodeGrayscale(corrupted, out _, out _));

        byte[] interlaced = BuildPng(1, 1, bitDepth: 8, colorType: 0, new byte[] { 0, 0 }, interlace: 1);
        FormatException error = Assert.Throws<FormatException>(() => PngCodec.DecodeGrayscale(interlaced, out _, out _));
        Assert.Contains("Interlaced", error.Message);
    }

    private static float Luminance(int r, int g, int b) => 0.299f * r + 0.587f * g + 0.114f * b;

    private static int Paeth(int a, int b, int c)
    {
        int p = a + b - c;
        int pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
        return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
    }

    /// <summary>Собирает PNG из уже отфильтрованных строк: так тест не зависит от кодировщика.</summary>
    private static byte[] BuildPng(int width, int height, byte bitDepth, byte colorType, byte[] raw, params (string Type, byte[] Data)[] extra) =>
        BuildPng(width, height, bitDepth, colorType, raw, interlace: 0, extra);

    private static byte[] BuildPng(int width, int height, byte bitDepth, byte colorType, byte[] raw, byte interlace, params (string Type, byte[] Data)[] extra)
    {
        using var output = new MemoryStream();
        output.Write(new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 });

        byte[] header = new byte[13];
        BinaryPrimitives.WriteInt32BigEndian(header, width);
        BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(4), height);
        header[8] = bitDepth;
        header[9] = colorType;
        header[12] = interlace;
        WriteChunk(output, "IHDR", header);
        foreach ((string type, byte[] data) in extra) WriteChunk(output, type, data);

        using var compressed = new MemoryStream();
        using (var zlib = new ZLibStream(compressed, CompressionLevel.Fastest, leaveOpen: true)) zlib.Write(raw);
        WriteChunk(output, "IDAT", compressed.ToArray());
        WriteChunk(output, "IEND", Array.Empty<byte>());
        return output.ToArray();
    }

    private static void WriteChunk(Stream output, string type, byte[] data)
    {
        byte[] length = new byte[4];
        BinaryPrimitives.WriteInt32BigEndian(length, data.Length);
        output.Write(length);
        byte[] typeAndData = Encoding.ASCII.GetBytes(type).Concat(data).ToArray();
        output.Write(typeAndData);
        byte[] crc = new byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(crc, Crc32(typeAndData));
        output.Write(crc);
    }

    private static uint Crc32(byte[] data)
    {
        uint crc = 0xFFFFFFFFu;
        foreach (byte b in data)
        {
            crc ^= b;
            for (int k = 0; k < 8; k++) crc = (crc & 1) != 0 ? 0xEDB88320u ^ (crc >> 1) : crc >> 1;
        }

        return crc ^ 0xFFFFFFFFu;
    }
}
