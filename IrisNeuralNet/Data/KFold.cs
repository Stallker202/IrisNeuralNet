using System;

namespace IrisNeuralNet.Data;

/// <summary>Один блок перекрёстной проверки.</summary>
public readonly record struct Fold(int Index, Dataset Train, Dataset Validation);

/// <summary>
/// K-Fold перекрёстная проверка, как в методичке Boston Housing: выборка делится на K
/// последовательных блоков по rows / K строк; блок i — валидация, остальное — обучение.
/// Остаток (rows mod K) всегда остаётся в обучении. Порядок строк не меняется —
/// перемешивайте выборку заранее (DatasetSplitter уже возвращает перемешанные строки).
/// </summary>
public static class KFold
{
    public static Fold[] Split(Dataset dataset, int k)
    {
        if (k < 2) throw new ArgumentOutOfRangeException(nameof(k), k, "K-Fold requires at least 2 folds.");

        int foldSize = dataset.RowCount / k;
        if (foldSize == 0) throw new ArgumentException($"Dataset has fewer rows than folds ({k}).", nameof(dataset));

        var folds = new Fold[k];
        var trainRows = new int[dataset.RowCount - foldSize];
        for (int i = 0; i < k; i++)
        {
            int validationStart = i * foldSize;
            int count = 0;
            for (int row = 0; row < dataset.RowCount; row++)
            {
                if (row < validationStart || row >= validationStart + foldSize) trainRows[count++] = row;
            }

            folds[i] = new Fold(i, dataset.Select(trainRows), dataset.Slice(validationStart, foldSize));
        }

        return folds;
    }
}
