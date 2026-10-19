namespace IrisNeuralNet.Lab.Reporting;

/// <summary>Согласование числительных: 1 эпоха, 2 эпохи, 5 эпох.</summary>
internal static class RussianPlural
{
    public static string Epochs(int count) => $"{count} {Choose(count, "эпоха", "эпохи", "эпох")}";

    public static string Choose(int count, string one, string few, string many)
    {
        int lastTwo = count % 100;
        int last = count % 10;
        if (lastTwo is >= 11 and <= 14) return many;
        return last switch
        {
            1 => one,
            >= 2 and <= 4 => few,
            _ => many,
        };
    }
}
