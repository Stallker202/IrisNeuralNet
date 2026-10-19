using System;

namespace IrisNeuralNet.Core;

internal sealed class BinaryTask : ClassificationTask
{
    private const float DecisionThreshold = 0.5f;

    public override string Description => "binary (sigmoid + binary CE)";

    public override Metric Metric => Metric.BinaryAccuracy;

    public override int OutputDim(int classCount) =>
        classCount == 2 ? 1 : throw new ArgumentOutOfRangeException(nameof(classCount), "Binary task requires exactly 2 classes.");

    public override ActivationFunction CreateOutputActivation() => new SigmoidActivation();

    public override ILossFunction CreateLoss() => new BinaryCrossEntropyLoss();

    /// <summary>Для sigmoid + BCE one-hot не нужен: метка 0/1 и есть цель.</summary>
    public override float[] EncodeTargets(ReadOnlySpan<float> labels, int classCount)
    {
        OutputDim(classCount);
        var targets = new float[labels.Length];
        for (int i = 0; i < labels.Length; i++)
        {
            targets[i] = ToClassIndex(labels[i], classCount);
        }

        return targets;
    }

    public override int PredictClass(ReadOnlySpan<float> output, out float confidence)
    {
        float probability = output[0];
        bool positive = probability >= DecisionThreshold;
        confidence = positive ? probability : 1f - probability;
        return positive ? 1 : 0;
    }
}
