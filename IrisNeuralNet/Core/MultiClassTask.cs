using System;
using IrisNeuralNet.Data;

namespace IrisNeuralNet.Core;

internal sealed class MultiClassTask : ClassificationTask
{
    public override string Description => "multi-class (softmax + categorical CE)";

    public override Metric Metric => Metric.Accuracy;

    public override int OutputDim(int classCount) =>
        classCount >= 2 ? classCount : throw new ArgumentOutOfRangeException(nameof(classCount));

    public override ActivationFunction CreateOutputActivation() => new SoftmaxActivation();

    public override ILossFunction CreateLoss() => new CategoricalCrossEntropyLoss();

    public override float[] EncodeTargets(ReadOnlySpan<float> labels, int classCount)
    {
        int outputDim = OutputDim(classCount);
        var classes = new int[labels.Length];
        for (int i = 0; i < labels.Length; i++)
        {
            classes[i] = ToClassIndex(labels[i], outputDim);
        }

        return OneHotEncoder.Encode(classes, classCount);
    }

    public override int PredictClass(ReadOnlySpan<float> output, out float confidence)
    {
        int best = Metrics.ArgMax(output);
        confidence = output[best];
        return best;
    }
}
