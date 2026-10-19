using System;

namespace IrisNeuralNet.Core;

/// <summary>
/// Гиперпараметры одного вызова <see cref="Trainer.Fit(ReadOnlySpan{float}, ReadOnlySpan{float}, in TrainingOptions, System.Threading.CancellationToken)"/>
/// (аналог аргументов model.fit: epochs, batch_size, плюс seed перемешивания и early stopping).
/// </summary>
/// <param name="EarlyStoppingPatience">0 — без ранней остановки; иначе число эпох без улучшения val_loss.</param>
public readonly record struct TrainingOptions(int Epochs, int BatchSize, int Seed = 0, int EarlyStoppingPatience = 0)
{
    internal void Validate()
    {
        if (Epochs <= 0) throw new ArgumentOutOfRangeException(nameof(Epochs), Epochs, "Epochs must be positive.");
        if (BatchSize <= 0) throw new ArgumentOutOfRangeException(nameof(BatchSize), BatchSize, "Batch size must be positive.");
        if (EarlyStoppingPatience < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(EarlyStoppingPatience), EarlyStoppingPatience, "Patience must not be negative.");
        }
    }
}
