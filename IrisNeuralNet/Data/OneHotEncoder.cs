using System;

namespace IrisNeuralNet.Data;

/// <summary>
/// One-hot кодирование индексов классов для categorical cross-entropy:
/// строка [0, 0, 1, ...] с единицей в позиции класса.
/// </summary>
public static class OneHotEncoder
{
    /// <summary>
    /// Кодирует индексы классов в новую one-hot матрицу [classes.Length x classCount].
    /// </summary>
    public static float[] Encode(ReadOnlySpan<int> classes, int classCount)
    {
        if (classCount <= 0) throw new ArgumentOutOfRangeException(nameof(classCount));

        var oneHot = new float[classes.Length * classCount];
        Encode(classes, classCount, oneHot);
        return oneHot;
    }

    /// <summary>
    /// Кодирует индексы классов в буфер вызывающего: ноль аллокаций.
    /// destination должен иметь длину classes.Length * classCount.
    /// </summary>
    public static void Encode(ReadOnlySpan<int> classes, int classCount, Span<float> destination)
    {
        if (classCount <= 0) throw new ArgumentOutOfRangeException(nameof(classCount));
        if (destination.Length != classes.Length * classCount)
        {
            throw new ArgumentException(
                "Destination length must equal classes.Length * classCount.",
                nameof(destination));
        }

        destination.Clear();
        for (int i = 0; i < classes.Length; i++)
        {
            int cls = classes[i];
            if ((uint)cls >= (uint)classCount)
            {
                throw new ArgumentException($"Class index {cls} is outside [0, {classCount}).", nameof(classes));
            }

            destination[i * classCount + cls] = 1f;
        }
    }
}