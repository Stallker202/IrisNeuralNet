using System;

namespace IrisNeuralNet.Core;

/// <summary>Перемешивание Фишера–Йетса; общее для Trainer, DatasetSplitter и K-Fold.</summary>
internal static class Shuffler
{
    public static void Shuffle(Span<int> order, Random random)
    {
        if (random is null) throw new ArgumentNullException(nameof(random));

        for (int i = order.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }
    }

    /// <summary>Возвращает перестановку 0..count-1, детерминированную по seed.</summary>
    public static int[] Permutation(int count, int seed)
    {
        var order = new int[count];
        for (int i = 0; i < count; i++) order[i] = i;
        Shuffle(order, new Random(seed));
        return order;
    }
}
