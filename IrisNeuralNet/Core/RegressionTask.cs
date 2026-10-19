using System;

namespace IrisNeuralNet.Core;

internal sealed class RegressionTask : TrainingTask
{
    public override string Description => "regression (linear + MSE)";

    public override Metric Metric => Metric.MeanAbsoluteError;

    public override bool IsClassification => false;

    public override int OutputDim(int classCount) => 1;

    public override ActivationFunction CreateOutputActivation() => new LinearActivation();

    public override ILossFunction CreateLoss() => new MeanSquaredErrorLoss();

    public override float[] EncodeTargets(ReadOnlySpan<float> labels, int classCount) => labels.ToArray();
}
