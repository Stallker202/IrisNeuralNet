using System;
using IrisNeuralNet.Core;

namespace IrisNeuralNet.Data;

/// <summary>Детерминированное (по seed) случайное разбиение на train/validation.</summary>
public static class DatasetSplitter
{
    /// <summary>
    /// Размер обучающей части — как в Keras: floor(rows · (1 − fraction)).
    /// Для Boston Housing (506 строк, 0.2) это даёт ровно 404 / 102, как в методичке.
    /// В отличие от validation_split в Keras строки перемешиваются: иначе в валидацию
    /// отсортированных iris.csv / sonar.csv попал бы один класс.
    /// </summary>
    public static (Dataset Train, Dataset Validation) Split(Dataset dataset, float validationFraction, int seed)
    {
        if (!(validationFraction > 0f && validationFraction < 1f))
        {
            throw new ArgumentOutOfRangeException(nameof(validationFraction));
        }

        int rowCount = dataset.RowCount;
        // Небольшой допуск компенсирует двоичную неточность float: 150 · (1 − 0.2f) должно дать 120, а не 119.
        int trainCount = (int)Math.Floor(rowCount * (1.0 - validationFraction) + 1e-6);
        if (trainCount <= 0 || trainCount >= rowCount)
        {
            throw new ArgumentException("Dataset is too small for the requested split.", nameof(dataset));
        }

        int[] order = Shuffler.Permutation(rowCount, seed);
        return (dataset.Select(order.AsSpan(0, trainCount)), dataset.Select(order.AsSpan(trainCount)));
    }
}
