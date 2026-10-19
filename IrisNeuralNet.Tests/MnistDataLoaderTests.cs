using System;
using System.Buffers.Binary;
using System.IO;
using System.IO.Compression;
using IrisNeuralNet.Data;
using Xunit;

namespace IrisNeuralNet.Tests;

public sealed class MnistDataLoaderTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "mnist-" + Path.GetRandomFileName());

    public MnistDataLoaderTests() => Directory.CreateDirectory(_directory);

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    [Fact]
    public void Load_ReadsGzippedIdxAndFlattensImages()
    {
        string images = WriteImages("img.gz", count: 2, compress: true);
        string labels = WriteLabels("lbl.gz", new byte[] { 7, 2 }, compress: true);

        Dataset dataset = MnistDataLoader.Load(images, labels);

        Assert.Equal(2, dataset.RowCount);
        Assert.Equal(784, dataset.FeatureCount);
        Assert.Equal(10, dataset.ClassCount);
        Assert.Equal(new float[] { 7f, 2f }, dataset.Labels.ToArray());
        Assert.Equal(PixelValue(1, 3), dataset.Row(1)[3]);
    }

    [Fact]
    public void LoadDirectory_AcceptsUncompressedStandardNames()
    {
        WriteImages("train-images-idx3-ubyte", 3, compress: false);
        WriteLabels("train-labels-idx1-ubyte", new byte[] { 0, 1, 2 }, compress: false);
        WriteImages("t10k-images-idx3-ubyte", 1, compress: false);
        WriteLabels("t10k-labels-idx1-ubyte", new byte[] { 9 }, compress: false);

        var (train, test) = MnistDataLoader.LoadDirectory(_directory);

        Assert.True(MnistDataLoader.IsAvailable(_directory));
        Assert.Equal(3, train.RowCount);
        Assert.Equal(9f, test.Labels[0]);
    }

    [Fact]
    public void Load_Throws_OnBadMagic()
    {
        string images = Path.Combine(_directory, "bad");
        File.WriteAllBytes(images, new byte[16]);
        string labels = WriteLabels("lbl", new byte[] { 1 }, compress: false);

        Assert.Throws<FormatException>(() => MnistDataLoader.Load(images, labels));
    }

    [Fact]
    public void Load_Throws_OnTruncatedPixels()
    {
        string images = WriteImages("img", count: 2, compress: false);
        byte[] bytes = File.ReadAllBytes(images);
        File.WriteAllBytes(images, bytes.AsSpan(0, bytes.Length - 100).ToArray());
        string labels = WriteLabels("lbl", new byte[] { 1, 2 }, compress: false);

        Assert.Throws<FormatException>(() => MnistDataLoader.Load(images, labels));
    }

    [Fact]
    public void Load_Throws_WhenCountsDiffer()
    {
        string images = WriteImages("img", count: 2, compress: false);
        string labels = WriteLabels("lbl", new byte[] { 1 }, compress: false);

        Assert.Throws<FormatException>(() => MnistDataLoader.Load(images, labels));
    }

    private static float PixelValue(int image, int pixel) => (image * 31 + pixel) % 256;

    private string WriteImages(string name, int count, bool compress)
    {
        using var buffer = new MemoryStream();
        WriteInt(buffer, 0x00000803);
        WriteInt(buffer, count);
        WriteInt(buffer, 28);
        WriteInt(buffer, 28);
        for (int i = 0; i < count; i++)
        {
            for (int p = 0; p < 784; p++) buffer.WriteByte((byte)PixelValue(i, p));
        }

        return Save(name, buffer.ToArray(), compress);
    }

    private string WriteLabels(string name, byte[] labels, bool compress)
    {
        using var buffer = new MemoryStream();
        WriteInt(buffer, 0x00000801);
        WriteInt(buffer, labels.Length);
        buffer.Write(labels);
        return Save(name, buffer.ToArray(), compress);
    }

    private string Save(string name, byte[] content, bool compress)
    {
        string path = Path.Combine(_directory, name);
        using FileStream file = File.Create(path);
        if (compress)
        {
            using var gzip = new GZipStream(file, CompressionLevel.Fastest);
            gzip.Write(content);
        }
        else
        {
            file.Write(content);
        }

        return path;
    }

    private static void WriteInt(Stream stream, int value)
    {
        Span<byte> bytes = stackalloc byte[4];
        BinaryPrimitives.WriteInt32BigEndian(bytes, value);
        stream.Write(bytes);
    }
}
