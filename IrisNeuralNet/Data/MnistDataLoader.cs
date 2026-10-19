using System;
using System.Buffers.Binary;
using System.IO;
using System.IO.Compression;

namespace IrisNeuralNet.Data;

/// <summary>
/// MNIST в оригинальном формате IDX (как на сайте LeCun и в зеркалах; .gz или распакованный).
/// Изображение 28×28 «вытягивается» по строкам в вектор из 784 признаков (аналог Flatten),
/// пиксели остаются в исходной шкале 0..255 — нормализация выполняется отдельным шагом.
/// </summary>
public static class MnistDataLoader
{
    public const int ImageSide = 28;
    public const int FeatureCount = ImageSide * ImageSide;
    public const int ClassCount = 10;

    public const string TrainImagesFile = "train-images-idx3-ubyte.gz";
    public const string TrainLabelsFile = "train-labels-idx1-ubyte.gz";
    public const string TestImagesFile = "t10k-images-idx3-ubyte.gz";
    public const string TestLabelsFile = "t10k-labels-idx1-ubyte.gz";

    private const int ImagesMagic = 0x00000803;
    private const int LabelsMagic = 0x00000801;

    /// <summary>Читает стандартную пару train (60000) / test (10000) из папки.</summary>
    public static (Dataset Train, Dataset Test) LoadDirectory(string directory)
    {
        if (directory is null) throw new ArgumentNullException(nameof(directory));

        return (
            Load(ResolvePath(directory, TrainImagesFile), ResolvePath(directory, TrainLabelsFile)),
            Load(ResolvePath(directory, TestImagesFile), ResolvePath(directory, TestLabelsFile)));
    }

    /// <summary>Истина, если в папке есть все четыре файла (.gz или распакованные).</summary>
    public static bool IsAvailable(string directory) =>
        Directory.Exists(directory)
        && Exists(directory, TrainImagesFile) && Exists(directory, TrainLabelsFile)
        && Exists(directory, TestImagesFile) && Exists(directory, TestLabelsFile);

    public static Dataset Load(string imagesPath, string labelsPath)
    {
        byte[] images = ReadIdx(imagesPath, ImagesMagic, out int imageCount, out int rows, out int columns);
        byte[] labels = ReadIdx(labelsPath, LabelsMagic, out int labelCount, out _, out _);

        if (rows != ImageSide || columns != ImageSide)
        {
            throw new FormatException($"'{imagesPath}': expected {ImageSide}x{ImageSide} images, got {rows}x{columns}.");
        }

        if (imageCount != labelCount)
        {
            throw new FormatException($"Image count {imageCount} does not match label count {labelCount}.");
        }

        var features = new float[images.Length];
        for (int i = 0; i < images.Length; i++) features[i] = images[i];

        var targets = new float[labels.Length];
        for (int i = 0; i < labels.Length; i++)
        {
            if (labels[i] >= ClassCount) throw new FormatException($"'{labelsPath}': label {labels[i]} at {i} is not a digit.");
            targets[i] = labels[i];
        }

        return new Dataset(features, targets, FeatureCount, ClassCount);
    }

    /// <summary>
    /// IDX: big-endian magic (тип + размерность), размеры по каждой оси, затем сырые байты.
    /// Для файла меток rows/columns = 1.
    /// </summary>
    private static byte[] ReadIdx(string path, int expectedMagic, out int count, out int rows, out int columns)
    {
        using Stream stream = OpenMaybeCompressed(path);
        Span<byte> header = stackalloc byte[4];

        int magic = ReadInt32(stream, header, path);
        if (magic != expectedMagic)
        {
            throw new FormatException($"'{path}': bad IDX magic 0x{magic:X8}, expected 0x{expectedMagic:X8}.");
        }

        count = ReadInt32(stream, header, path);
        bool isImages = expectedMagic == ImagesMagic;
        rows = isImages ? ReadInt32(stream, header, path) : 1;
        columns = isImages ? ReadInt32(stream, header, path) : 1;
        if (count <= 0 || rows <= 0 || columns <= 0)
        {
            throw new FormatException($"'{path}': invalid IDX dimensions {count}x{rows}x{columns}.");
        }

        var data = new byte[checked(count * rows * columns)];
        ReadExactly(stream, data, path);
        return data;
    }

    private static int ReadInt32(Stream stream, Span<byte> buffer, string path)
    {
        ReadExactly(stream, buffer, path);
        return BinaryPrimitives.ReadInt32BigEndian(buffer);
    }

    private static void ReadExactly(Stream stream, Span<byte> buffer, string path)
    {
        try
        {
            stream.ReadExactly(buffer);
        }
        catch (EndOfStreamException e)
        {
            throw new FormatException($"'{path}': IDX file is truncated.", e);
        }
    }

    private static Stream OpenMaybeCompressed(string path)
    {
        var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1 << 16);
        return path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase)
            ? new GZipStream(file, CompressionMode.Decompress)
            : file;
    }

    /// <summary>Предпочитает .gz; если его нет — распакованный файл с тем же именем.</summary>
    private static string ResolvePath(string directory, string gzFileName)
    {
        string gz = Path.Combine(directory, gzFileName);
        if (File.Exists(gz)) return gz;

        string raw = Path.Combine(directory, Path.GetFileNameWithoutExtension(gzFileName));
        if (File.Exists(raw)) return raw;

        throw new FileNotFoundException($"MNIST file '{gzFileName}' not found in '{directory}'.", gz);
    }

    private static bool Exists(string directory, string gzFileName) =>
        File.Exists(Path.Combine(directory, gzFileName))
        || File.Exists(Path.Combine(directory, Path.GetFileNameWithoutExtension(gzFileName)));
}
