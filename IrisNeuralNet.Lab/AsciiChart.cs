using System;
using System.Text;
using System.Collections.Generic;

namespace IrisNeuralNet.Lab
{
    internal static class AsciiChart
    {
        private const int Height = 12;
        private const int Width = 60;

        public static void Print(
            string title,
            IReadOnlyList<float> first,
            IReadOnlyList<float> second,
            string firstLabel,
            string secondLabel)
        {
            if (first.Count == 0 || second.Count == 0)
            {
                return;
            }

            float min = MathF.Min(Min(first), Min(second));
            float max = MathF.Max(Max(first), Max(second));
            if (max - min < 1e-9f)
            {
                max = min + 1f;
            }

            char[,] grid = new char[Height, Width];
            for (int row = 0; row < Height; row++)
            {
                for (int col = 0; col < Width; col++)
                {
                    grid[row, col] = ' ';
                }
            }

            Plot(grid, first, min, max, '#');
            Plot(grid, second, min, max, '*');

            Console.WriteLine();
            Console.WriteLine($"{title}:  # = {firstLabel},  * = {secondLabel}");
            for (int row = 0; row < Height; row++)
            {
                string axis = row == 0 ? $"{max,9:F4} |" : row == Height - 1 ? $"{min,9:F4} |" : "          |";
                var line = new StringBuilder(Width);
                for (int col = 0; col < Width; col++)
                {
                    line.Append(grid[row, col]);
                }

                Console.WriteLine(axis + line);
            }
        }

        private static void Plot(char[,] grid, IReadOnlyList<float> series, float min, float max, char marker)
        {
            for (int i = 0; i < series.Count; i++)
            {
                int col = series.Count == 1 ? 0 : (int)Math.Round(i * (Width - 1) / (double)(series.Count - 1));
                float clamped = MathF.Min(MathF.Max(series[i], min), max);
                int row = (int)Math.Round((max - clamped) / (max - min) * (Height - 1));
                grid[row, col] = marker;
            }
        }

        private static float Min(IReadOnlyList<float> values)
        {
            float result = float.MaxValue;
            for (int i = 0; i < values.Count; i++)
            {
                result = MathF.Min(result, values[i]);
            }

            return result;
        }

        private static float Max(IReadOnlyList<float> values)
        {
            float result = float.MinValue;
            for (int i = 0; i < values.Count; i++)
            {
                result = MathF.Max(result, values[i]);
            }

            return result;
        }
    }
}
