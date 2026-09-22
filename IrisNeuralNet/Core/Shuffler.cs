using System;

namespace IrisNeuralNet.Core
{
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
    }
}
