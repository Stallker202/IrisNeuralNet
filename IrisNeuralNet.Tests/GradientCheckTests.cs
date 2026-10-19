using System;
using IrisNeuralNet.Core;
using Xunit;

namespace IrisNeuralNet.Tests;

/// <summary>
/// Сверка аналитических градиентов backprop с численными центральными разностями
/// (dL/dw ≈ [L(w + ε) − L(w − ε)] / 2ε) для всех трёх связок «выход + потери».
/// Скрытый слой — сигмоида: гладкая функция без излома ReLU, поэтому разности корректны.
/// Заодно проверяется, что пропуск dX у первого слоя не ломает его dW.
/// </summary>
public sealed class GradientCheckTests
{
    private const float Epsilon = 1e-3f;

    [Theory]
    [InlineData("multiclass")]
    [InlineData("binary")]
    [InlineData("regression")]
    public void AnalyticGradients_MatchFiniteDifferences(string taskName)
    {
        TrainingTask task = taskName switch
        {
            "multiclass" => TrainingTask.MultiClass,
            "binary" => TrainingTask.Binary,
            _ => TrainingTask.Regression,
        };
        int classCount = taskName == "multiclass" ? 3 : taskName == "binary" ? 2 : 0;
        int outputDim = task.OutputDim(classCount);
        var random = new Random(5);

        var network = new NeuralNetwork(new[]
        {
            new DenseLayer(3, 4, new SigmoidActivation(), seed: 11),
            new DenseLayer(4, outputDim, task.CreateOutputActivation(), seed: 12),
        });
        ILossFunction loss = task.CreateLoss();

        const int rows = 5;
        float[] x = new float[rows * 3];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(random.NextDouble() * 2 - 1);
        float[] labels = new float[rows];
        for (int i = 0; i < rows; i++) labels[i] = classCount > 0 ? i % classCount : (float)random.NextDouble() * 3f;
        float[] y = task.EncodeTargets(labels, classCount);

        float[] predicted = new float[rows * outputDim];
        float[] outputGradient = new float[rows * outputDim];
        network.Forward(x, predicted, rows);
        loss.Backward(predicted, y, outputDim, outputGradient);
        network.Backward(outputGradient, rows);

        foreach (DenseLayer layer in network.Layers)
        {
            float[] analytic = layer.WeightGradients.ToArray();
            for (int i = 0; i < analytic.Length; i++)
            {
                float original = layer.Weights[i];
                layer.Weights[i] = original + Epsilon;
                float plus = LossOf(network, loss, x, y, rows);
                layer.Weights[i] = original - Epsilon;
                float minus = LossOf(network, loss, x, y, rows);
                layer.Weights[i] = original;

                float numeric = (plus - minus) / (2 * Epsilon);
                Assert.True(
                    MathF.Abs(numeric - analytic[i]) <= 2e-3f + 2e-2f * MathF.Abs(numeric),
                    $"{taskName}: weight {i}: analytic {analytic[i]}, numeric {numeric}");
            }
        }
    }

    private static float LossOf(NeuralNetwork network, ILossFunction loss, float[] x, float[] y, int rows)
    {
        float[] output = new float[rows * network.OutputDim];
        network.Infer(x, output, rows);
        return loss.Compute(output, y, network.OutputDim);
    }
}
