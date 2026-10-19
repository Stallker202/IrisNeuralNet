using System;
using System.Buffers.Binary;
using System.IO;
using System.IO.Compression;
using System.Text;

namespace IrisNeuralNet.Lab.Imaging;

/// <summary>
/// PNG без зависимостей: запись 8-битных изображений (цифры MNIST в отчётах и превью)
/// и чтение типичных PNG в оттенки серого (своя цифра для --image). Сжатие — встроенный
/// в .NET ZLibStream, поэтому работает одинаково на Windows, macOS и Linux.
/// Поддерживаются все типы цвета, глубина 1–16 бит, без чересстрочной развёртки (Adam7).
/// </summary>
internal static class PngCodec
{
    public const int MaxDimension = 16384;
    private const long MaxRawBytes = 256L * 1024 * 1024;

    private static readonly byte[] Signature = { 137, 80, 78, 71, 13, 10, 26, 10 };
    private static readonly uint[] CrcTable = CreateCrcTable();

    public static byte[] EncodeGray(ReadOnlySpan<byte> pixels, int width, int height) =>
        Encode(pixels, width, height, colorType: 0, channels: 1);

    public static byte[] EncodeRgba(ReadOnlySpan<byte> pixels, int width, int height) =>
        Encode(pixels, width, height, colorType: 6, channels: 4);

    public static string ToDataUri(byte[] png) => "data:image/png;base64," + Convert.ToBase64String(png);

    /// <summary>
    /// Декодирует PNG в яркость 0..255 (0 — чёрный). Прозрачность смешивается с белым фоном,
    /// как у картинки «на бумаге»; цвет — по формуле Rec. 601.
    /// </summary>
    public static float[] DecodeGrayscale(ReadOnlySpan<byte> png, out int width, out int height)
    {
        if (png.Length < Signature.Length || !png[..Signature.Length].SequenceEqual(Signature))
        {
            throw new FormatException("Not a PNG file.");
        }

        var header = default(Header);
        byte[] palette = Array.Empty<byte>();
        byte[] transparency = Array.Empty<byte>();
        using var compressed = new MemoryStream();
        int offset = Signature.Length;
        bool ended = false;
        while (!ended)
        {
            if (offset + 12 > png.Length) throw new FormatException("PNG is truncated.");

            uint declared = BinaryPrimitives.ReadUInt32BigEndian(png[offset..]);
            if (offset + 12L + declared > png.Length) throw new FormatException("PNG chunk is truncated.");

            int length = (int)declared;

            ReadOnlySpan<byte> typeAndData = png.Slice(offset + 4, 4 + length);
            uint expectedCrc = BinaryPrimitives.ReadUInt32BigEndian(png[(offset + 8 + length)..]);
            if (Crc32(typeAndData) != expectedCrc) throw new FormatException("PNG chunk CRC mismatch.");

            string type = Encoding.ASCII.GetString(typeAndData[..4]);
            ReadOnlySpan<byte> data = typeAndData[4..];
            switch (type)
            {
                case "IHDR": header = Header.Parse(data); break;
                case "PLTE": palette = data.ToArray(); break;
                case "tRNS": transparency = data.ToArray(); break;
                case "IDAT": compressed.Write(data); break;
                case "IEND": ended = true; break;
            }

            offset += 12 + length;
        }

        if (header.Width == 0) throw new FormatException("PNG has no IHDR chunk.");
        if (header.ColorType == 3 && palette.Length == 0) throw new FormatException("Palette PNG has no PLTE chunk.");

        width = header.Width;
        height = header.Height;
        byte[] raw = Inflate(compressed, header);
        return ToGrayscale(raw, header, palette, transparency);
    }

    private static byte[] Encode(ReadOnlySpan<byte> pixels, int width, int height, byte colorType, int channels)
    {
        if (width <= 0 || width > MaxDimension) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0 || height > MaxDimension) throw new ArgumentOutOfRangeException(nameof(height));
        if (pixels.Length != width * height * channels)
        {
            throw new ArgumentException($"Expected {width * height * channels} bytes for {width}x{height}.", nameof(pixels));
        }

        using var output = new MemoryStream();
        output.Write(Signature);

        Span<byte> ihdr = stackalloc byte[13];
        BinaryPrimitives.WriteInt32BigEndian(ihdr, width);
        BinaryPrimitives.WriteInt32BigEndian(ihdr[4..], height);
        ihdr[8] = 8;
        ihdr[9] = colorType;
        WriteChunk(output, "IHDR", ihdr);

        using var compressed = new MemoryStream();
        using (var zlib = new ZLibStream(compressed, CompressionLevel.Optimal, leaveOpen: true))
        {
            int stride = width * channels;
            for (int y = 0; y < height; y++)
            {
                zlib.WriteByte(0); // фильтр None
                zlib.Write(pixels.Slice(y * stride, stride));
            }
        }

        WriteChunk(output, "IDAT", compressed.ToArray());
        WriteChunk(output, "IEND", ReadOnlySpan<byte>.Empty);
        return output.ToArray();
    }

    private static byte[] Inflate(MemoryStream compressed, Header header)
    {
        long rawLength = (long)header.Height * (header.Stride + 1);
        if (rawLength > MaxRawBytes) throw new FormatException("PNG is too large.");

        var raw = new byte[rawLength];
        compressed.Position = 0;
        using var zlib = new ZLibStream(compressed, CompressionMode.Decompress);
        try
        {
            zlib.ReadExactly(raw);
        }
        catch (EndOfStreamException e)
        {
            throw new FormatException("PNG image data is truncated.", e);
        }
        catch (InvalidDataException e)
        {
            throw new FormatException("PNG image data is corrupt.", e);
        }

        Unfilter(raw, header);
        return raw;
    }

    /// <summary>Снимает построчные фильтры PNG (None, Sub, Up, Average, Paeth) на месте.</summary>
    private static void Unfilter(byte[] raw, Header header)
    {
        int stride = header.Stride;
        int bpp = Math.Max(1, header.BitsPerPixel / 8);
        for (int y = 0; y < header.Height; y++)
        {
            int row = y * (stride + 1);
            byte filter = raw[row];
            Span<byte> current = raw.AsSpan(row + 1, stride);
            ReadOnlySpan<byte> prior = y == 0 ? ReadOnlySpan<byte>.Empty : raw.AsSpan(row - stride, stride);
            for (int i = 0; i < stride; i++)
            {
                int left = i >= bpp ? current[i - bpp] : 0;
                int up = prior.IsEmpty ? 0 : prior[i];
                int upLeft = !prior.IsEmpty && i >= bpp ? prior[i - bpp] : 0;
                int predictor = filter switch
                {
                    0 => 0,
                    1 => left,
                    2 => up,
                    3 => (left + up) >> 1,
                    4 => Paeth(left, up, upLeft),
                    _ => throw new FormatException($"Unknown PNG filter {filter}."),
                };
                current[i] = (byte)(current[i] + predictor);
            }
        }
    }

    private static int Paeth(int a, int b, int c)
    {
        int p = a + b - c;
        int pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
        return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
    }

    private static float[] ToGrayscale(byte[] raw, Header header, byte[] palette, byte[] transparency)
    {
        var gray = new float[header.Width * header.Height];
        int maxSample = (1 << Math.Min(header.BitDepth, 8)) - 1;
        for (int y = 0; y < header.Height; y++)
        {
            ReadOnlySpan<byte> row = raw.AsSpan(y * (header.Stride + 1) + 1, header.Stride);
            for (int x = 0; x < header.Width; x++)
            {
                int r, g, b, a = 255;
                switch (header.ColorType)
                {
                    case 0:
                        r = g = b = Sample(row, header, x, 0) * 255 / maxSample;
                        break;
                    case 2:
                        r = Sample(row, header, x, 0);
                        g = Sample(row, header, x, 1);
                        b = Sample(row, header, x, 2);
                        break;
                    case 3:
                        int index = Sample(row, header, x, 0);
                        if (index * 3 + 2 >= palette.Length) throw new FormatException($"Palette index {index} is out of range.");
                        r = palette[index * 3];
                        g = palette[index * 3 + 1];
                        b = palette[index * 3 + 2];
                        if (index < transparency.Length) a = transparency[index];
                        break;
                    case 4:
                        r = g = b = Sample(row, header, x, 0);
                        a = Sample(row, header, x, 1);
                        break;
                    default:
                        r = Sample(row, header, x, 0);
                        g = Sample(row, header, x, 1);
                        b = Sample(row, header, x, 2);
                        a = Sample(row, header, x, 3);
                        break;
                }

                float alpha = a / 255f;
                float luminance = r == g && g == b ? r : 0.299f * r + 0.587f * g + 0.114f * b;
                gray[y * header.Width + x] = luminance * alpha + 255f * (1f - alpha);
            }
        }

        return gray;
    }

    /// <summary>Отсчёт канала: подбайтовые глубины распаковываются, у 16-битных берётся старший байт.</summary>
    private static int Sample(ReadOnlySpan<byte> row, Header header, int x, int channel)
    {
        int depth = header.BitDepth;
        if (depth == 8) return row[x * header.Channels + channel];
        if (depth == 16) return row[(x * header.Channels + channel) * 2];

        int bit = x * depth;
        int shift = 8 - depth - (bit & 7);
        return (row[bit >> 3] >> shift) & ((1 << depth) - 1);
    }

    private static void WriteChunk(Stream output, string type, ReadOnlySpan<byte> data)
    {
        Span<byte> buffer = stackalloc byte[4];
        BinaryPrimitives.WriteInt32BigEndian(buffer, data.Length);
        output.Write(buffer);

        byte[] typeAndData = new byte[4 + data.Length];
        Encoding.ASCII.GetBytes(type, typeAndData);
        data.CopyTo(typeAndData.AsSpan(4));
        output.Write(typeAndData);

        BinaryPrimitives.WriteUInt32BigEndian(buffer, Crc32(typeAndData));
        output.Write(buffer);
    }

    private static uint Crc32(ReadOnlySpan<byte> data)
    {
        uint crc = 0xFFFFFFFFu;
        foreach (byte b in data) crc = CrcTable[(crc ^ b) & 0xFF] ^ (crc >> 8);
        return crc ^ 0xFFFFFFFFu;
    }

    private static uint[] CreateCrcTable()
    {
        var table = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++) c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[n] = c;
        }

        return table;
    }

    private readonly record struct Header(int Width, int Height, int BitDepth, int ColorType, int Channels)
    {
        public int BitsPerPixel => BitDepth * Channels;
        public int Stride => (Width * BitsPerPixel + 7) / 8;

        public static Header Parse(ReadOnlySpan<byte> data)
        {
            if (data.Length < 13) throw new FormatException("PNG IHDR is too short.");

            int width = BinaryPrimitives.ReadInt32BigEndian(data);
            int height = BinaryPrimitives.ReadInt32BigEndian(data[4..]);
            int depth = data[8];
            int colorType = data[9];
            if (width <= 0 || height <= 0 || width > MaxDimension || height > MaxDimension)
            {
                throw new FormatException($"PNG size {width}x{height} is not supported.");
            }

            if (data[12] != 0) throw new FormatException("Interlaced PNG (Adam7) is not supported — save the image without interlacing.");

            int channels = colorType switch
            {
                0 or 3 => 1,
                2 => 3,
                4 => 2,
                6 => 4,
                _ => throw new FormatException($"Unknown PNG color type {colorType}."),
            };
            bool validDepth = colorType switch
            {
                0 => depth is 1 or 2 or 4 or 8 or 16,
                3 => depth is 1 or 2 or 4 or 8,
                _ => depth is 8 or 16,
            };
            if (!validDepth) throw new FormatException($"PNG bit depth {depth} is invalid for color type {colorType}.");

            return new Header(width, height, depth, colorType, channels);
        }
    }
}
