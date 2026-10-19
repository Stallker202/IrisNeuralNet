using System;
using System.Buffers.Text;
using System.IO;
using System.Text;

namespace IrisNeuralNet.Data;

/// <summary>
/// Парсер CSV без заголовка: N числовых признаков + метка в последней колонке.
/// Метка — строковое имя класса (классификация) или число (регрессия).
/// Без строк-посредников: файл читается как byte[] и разбирается Utf8Parser'ом.
/// </summary>
public static class CsvDatasetLoader
{
    /// <summary>Разбор метки строки: байты последней колонки → float.</summary>
    private delegate float LabelParser(ReadOnlySpan<byte> label, int rowIndex);

    /// <param name="labelNames">Имена классов; индекс имени — номер класса.</param>
    public static Dataset LoadClassification(string path, int featureCount, string[] labelNames)
    {
        if (labelNames is null || labelNames.Length < 2)
        {
            throw new ArgumentException("At least two label names are required.", nameof(labelNames));
        }

        byte[][] labelBytes = new byte[labelNames.Length][];
        for (int i = 0; i < labelNames.Length; i++)
        {
            labelBytes[i] = Encoding.UTF8.GetBytes(labelNames[i]);
        }

        return Load(path, featureCount, labelNames.Length, (label, row) => MatchClass(label, labelBytes, row));
    }

    /// <summary>Последняя колонка — числовая цель регрессии.</summary>
    public static Dataset LoadRegression(string path, int featureCount) =>
        Load(path, featureCount, classCount: 0, ParseNumericLabel);

    private static Dataset Load(string path, int featureCount, int classCount, LabelParser parseLabel)
    {
        if (path is null) throw new ArgumentNullException(nameof(path));
        if (featureCount <= 0) throw new ArgumentOutOfRangeException(nameof(featureCount));

        byte[] bytes = File.ReadAllBytes(path);
        int rowCount = CountRows(bytes);
        if (rowCount == 0) throw new FormatException($"Dataset '{path}' is empty.");

        var features = new float[rowCount * featureCount];
        var labels = new float[rowCount];
        int position = 0;
        for (int row = 0; row < rowCount; row++)
        {
            position = SkipBlankLines(bytes, position);
            position = ParseRow(bytes, position, row, features.AsSpan(row * featureCount, featureCount), out ReadOnlySpan<byte> label);
            labels[row] = parseLabel(label, row);
        }

        return new Dataset(features, labels, featureCount, classCount);
    }

    private static int ParseRow(byte[] bytes, int start, int rowIndex, Span<float> features, out ReadOnlySpan<byte> label)
    {
        int position = start;
        for (int f = 0; f < features.Length; f++)
        {
            position = SkipWhitespace(bytes, position);
            if (!Utf8Parser.TryParse(bytes.AsSpan(position), out float value, out int consumed))
            {
                throw new FormatException($"Row {rowIndex + 1}: expected a number at offset {position}.");
            }

            features[f] = value;
            position = ExpectComma(bytes, position + consumed, rowIndex);
        }

        position = SkipWhitespace(bytes, position);
        int labelStart = position;
        while (position < bytes.Length && bytes[position] != (byte)'\n' && bytes[position] != (byte)'\r')
        {
            position++;
        }

        label = TrimEnd(bytes.AsSpan(labelStart, position - labelStart));

        if (position < bytes.Length && bytes[position] == (byte)'\r') position++;
        if (position < bytes.Length && bytes[position] == (byte)'\n') position++;
        return position;
    }

    private static float MatchClass(ReadOnlySpan<byte> label, byte[][] labelBytes, int rowIndex)
    {
        for (int i = 0; i < labelBytes.Length; i++)
        {
            if (label.SequenceEqual(labelBytes[i])) return i;
        }

        throw new FormatException($"Row {rowIndex + 1}: unknown class label '{Encoding.UTF8.GetString(label)}'.");
    }

    private static float ParseNumericLabel(ReadOnlySpan<byte> label, int rowIndex)
    {
        if (!Utf8Parser.TryParse(label, out float value, out int consumed) || consumed != label.Length)
        {
            throw new FormatException($"Row {rowIndex + 1}: expected a numeric target, got '{Encoding.UTF8.GetString(label)}'.");
        }

        return value;
    }

    private static int CountRows(byte[] bytes)
    {
        int rows = 0;
        bool inRow = false;
        foreach (byte b in bytes)
        {
            if (b == (byte)'\n')
            {
                if (inRow) rows++;
                inRow = false;
            }
            else if (b != (byte)'\r' && b != (byte)' ' && b != (byte)'\t')
            {
                inRow = true;
            }
        }

        if (inRow) rows++;
        return rows;
    }

    private static int SkipBlankLines(byte[] bytes, int position)
    {
        while (position < bytes.Length)
        {
            byte b = bytes[position];
            if (b == (byte)'\n' || b == (byte)'\r')
            {
                position++;
                continue;
            }

            if (b == (byte)' ' || b == (byte)'\t')
            {
                int probe = SkipWhitespace(bytes, position);
                if (probe >= bytes.Length || bytes[probe] == (byte)'\n' || bytes[probe] == (byte)'\r')
                {
                    position = probe;
                    continue;
                }
            }

            break;
        }

        return position;
    }

    private static int ExpectComma(byte[] bytes, int position, int rowIndex)
    {
        position = SkipWhitespace(bytes, position);
        if (position >= bytes.Length || bytes[position] != (byte)',')
        {
            throw new FormatException($"Row {rowIndex + 1}: expected ',' at offset {position}.");
        }

        return position + 1;
    }

    private static int SkipWhitespace(byte[] bytes, int position)
    {
        while (position < bytes.Length && (bytes[position] == (byte)' ' || bytes[position] == (byte)'\t'))
        {
            position++;
        }

        return position;
    }

    private static ReadOnlySpan<byte> TrimEnd(ReadOnlySpan<byte> label)
    {
        int end = label.Length;
        while (end > 0 && (label[end - 1] == (byte)' ' || label[end - 1] == (byte)'\t'))
        {
            end--;
        }

        return label.Slice(0, end);
    }
}
