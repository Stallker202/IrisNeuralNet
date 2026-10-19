using System;
using System.Collections.Generic;
using System.Globalization;

namespace IrisNeuralNet.Lab.Experiments;

/// <summary>Разбор и запись размеров скрытых слоёв: «512, 256» ↔ [512, 256]. Пусто или «-» — без скрытых слоёв.</summary>
internal static class ArchitectureParser
{
    public const int MaxNeurons = 4096;
    public const int MaxLayers = 8;

    private static readonly char[] Separators = { ',', ';', ' ', '→' };

    public static bool TryParse(string? text, out int[] hidden, out string error)
    {
        hidden = Array.Empty<int>();
        error = string.Empty;
        if (string.IsNullOrWhiteSpace(text) || text.Trim() == "-")
        {
            return true;
        }

        var sizes = new List<int>();
        foreach (string token in text.Replace("->", ",").Split(Separators, StringSplitOptions.RemoveEmptyEntries))
        {
            if (!int.TryParse(token, NumberStyles.None, CultureInfo.InvariantCulture, out int size) || size < 1 || size > MaxNeurons)
            {
                error = $"«{token}» — нужно целое от 1 до {MaxNeurons}";
                return false;
            }

            sizes.Add(size);
        }

        if (sizes.Count > MaxLayers)
        {
            error = $"не больше {MaxLayers} скрытых слоёв";
            return false;
        }

        hidden = sizes.ToArray();
        return true;
    }

    public static int[] Parse(string? text) =>
        TryParse(text, out int[] hidden, out string error) ? hidden : throw new FormatException(error);

    public static string Format(IReadOnlyList<int> hidden) =>
        hidden.Count == 0 ? "-" : string.Join(", ", hidden);

    /// <summary>Сетка «L слоёв по N нейронов» для sweep.</summary>
    public static int[] Uniform(int layers, int neurons)
    {
        var hidden = new int[layers];
        Array.Fill(hidden, neurons);
        return hidden;
    }
}
