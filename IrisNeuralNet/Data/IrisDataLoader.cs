using System;
using System.Buffers.Text;
using System.IO;
using System.Text;

namespace IrisNeuralNet.Data
{
    /// <summary>
    /// Читает iris.csv (формат UCI: 4 признака + метка класса, без заголовка)
    /// напрямую из UTF-8 байтов, без промежуточных строк.
    /// </summary>
    public static class IrisDataLoader
    {
        public const int FeatureCount = 4;
        public const int ClassCount = 3;

        private static readonly byte[][] ClassLabels =
        {
        Encoding.UTF8.GetBytes("Iris-setosa"),
        Encoding.UTF8.GetBytes("Iris-versicolor"),
        Encoding.UTF8.GetBytes("Iris-virginica"),
    };

        public static IrisDataset Load(string path)
        {
            if (path is null) throw new ArgumentNullException(nameof(path));

            byte[] bytes = File.ReadAllBytes(path);
            int rowCount = CountRows(bytes);
            if (rowCount == 0) throw new FormatException("Dataset is empty.");

            var features = new float[rowCount * FeatureCount];
            var classes = new int[rowCount];

            int position = 0;
            for (int row = 0; row < rowCount; row++)
            {
                position = SkipBlankLines(bytes, position);
                position = ParseRow(bytes, position, row, features, classes);
            }

            return new IrisDataset(features, classes, FeatureCount, ClassCount);
        }

        private static int CountRows(byte[] bytes)
        {
            int rows = 0;
            bool inRow = false;
            for (int i = 0; i < bytes.Length; i++)
            {
                byte b = bytes[i];
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

        private static int ParseRow(byte[] bytes, int start, int rowIndex, float[] features, int[] classes)
        {
            int position = start;
            for (int f = 0; f < FeatureCount; f++)
            {
                position = SkipWhitespace(bytes, position);
                if (!Utf8Parser.TryParse(bytes.AsSpan(position), out float value, out int consumed))
                {
                    throw new FormatException($"Row {rowIndex + 1}: expected a number at offset {position}.");
                }

                features[rowIndex * FeatureCount + f] = value;
                position += consumed;

                if (f < FeatureCount - 1)
                {
                    position = ExpectComma(bytes, position, rowIndex);
                }
            }

            position = ExpectComma(bytes, position, rowIndex);
            position = SkipWhitespace(bytes, position);

            int labelStart = position;
            while (position < bytes.Length && bytes[position] != (byte)'\n' && bytes[position] != (byte)'\r')
            {
                position++;
            }

            ReadOnlySpan<byte> label = TrimEnd(bytes.AsSpan(labelStart, position - labelStart));
            classes[rowIndex] = MatchClass(label, rowIndex);

            if (position < bytes.Length && bytes[position] == (byte)'\r') position++;
            if (position < bytes.Length && bytes[position] == (byte)'\n') position++;
            return position;
        }

        private static int ExpectComma(byte[] bytes, int position, int rowIndex)
        {
            if (position >= bytes.Length || bytes[position] != (byte)',')
            {
                throw new FormatException($"Row {rowIndex + 1}: expected ',' at offset {position}.");
            }

            return position + 1;
        }

        private static int MatchClass(ReadOnlySpan<byte> label, int rowIndex)
        {
            for (int i = 0; i < ClassLabels.Length; i++)
            {
                if (label.SequenceEqual(ClassLabels[i])) return i;
            }

            throw new FormatException($"Row {rowIndex + 1}: unknown class label '{Encoding.UTF8.GetString(label)}'.");
        }

        private static int SkipWhitespace(byte[] bytes, int position)
        {
            while (position < bytes.Length && (bytes[position] == (byte)' ' || bytes[position] == (byte)'\t'))
            {
                position++;
            }

            return position;
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
                    if (probe < bytes.Length && (bytes[probe] == (byte)'\n' || bytes[probe] == (byte)'\r'))
                    {
                        position = probe;
                        continue;
                    }
                }

                break;
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
}
