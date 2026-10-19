using System;

namespace IrisNeuralNet.Core;

/// <summary>
/// Тип задачи обучения (Strategy): задаёт выходной слой, функцию потерь, метрику
/// и кодирование меток. Добавление нового типа задачи — новый класс, без switch по коду.
/// </summary>
public abstract class TrainingTask
{
    /// <summary>Многоклассовая классификация: softmax + categorical cross-entropy + one-hot.</summary>
    public static readonly ClassificationTask MultiClass = new MultiClassTask();

    /// <summary>Бинарная классификация: 1 нейрон sigmoid + binary cross-entropy, метки 0/1.</summary>
    public static readonly ClassificationTask Binary = new BinaryTask();

    /// <summary>Регрессия: 1 линейный нейрон + MSE, метрика MAE.</summary>
    public static readonly TrainingTask Regression = new RegressionTask();

    protected TrainingTask()
    {
    }

    /// <summary>Короткое описание связки «выход + потери» для статусов и отчётов.</summary>
    public abstract string Description { get; }

    public abstract Metric Metric { get; }

    public abstract bool IsClassification { get; }

    /// <summary>Число нейронов выходного слоя для датасета с classCount классами.</summary>
    public abstract int OutputDim(int classCount);

    public abstract ActivationFunction CreateOutputActivation();

    public abstract ILossFunction CreateLoss();

    /// <summary>
    /// Превращает метки датасета (индекс класса или значение регрессии, по одной на строку)
    /// в целевую матрицу [rows x OutputDim(classCount)].
    /// </summary>
    public abstract float[] EncodeTargets(ReadOnlySpan<float> labels, int classCount);

    public override string ToString() => Description;
}

/// <summary>Классификация: дополнительно умеет превращать выход сети в решение «класс + уверенность».</summary>
public abstract class ClassificationTask : TrainingTask
{
    protected ClassificationTask()
    {
    }

    public sealed override bool IsClassification => true;

    /// <summary>Решение по выходу сети для одного образца.</summary>
    public abstract int PredictClass(ReadOnlySpan<float> output, out float confidence);

    /// <summary>Проверяет, что метка — целый индекс класса из [0, classCount).</summary>
    protected static int ToClassIndex(float label, int classCount)
    {
        int index = (int)label;
        if (index != label || (uint)index >= (uint)classCount)
        {
            throw new ArgumentException($"Label {label} is not a class index in [0, {classCount}).");
        }

        return index;
    }
}
