using System;

namespace IrisNeuralNet.Core;

/// <summary>Единственное место сборки архитектуры «вход → скрытые ReLU-слои → выход» по спецификации.</summary>
public static class NetworkFactory
{
    /// <param name="hiddenNeurons">Размеры скрытых слоёв (ReLU); пусто — сеть из одного выходного слоя.</param>
    public static NeuralNetwork Create(
        int inputDim, ReadOnlySpan<int> hiddenNeurons, int outputDim, ActivationFunction outputActivation, int seed)
    {
        if (outputActivation is null) throw new ArgumentNullException(nameof(outputActivation));

        var layers = new DenseLayer[hiddenNeurons.Length + 1];
        int input = inputDim;
        for (int i = 0; i < hiddenNeurons.Length; i++)
        {
            layers[i] = new DenseLayer(input, hiddenNeurons[i], new ReLUActivation(), seed: seed + i + 1);
            input = hiddenNeurons[i];
        }

        layers[^1] = new DenseLayer(input, outputDim, outputActivation, seed: seed + hiddenNeurons.Length + 1);
        return new NeuralNetwork(layers);
    }

    /// <summary>Выходной слой (размер и активация) определяется типом задачи.</summary>
    public static NeuralNetwork Create(
        int inputDim, ReadOnlySpan<int> hiddenNeurons, TrainingTask task, int classCount, int seed)
    {
        if (task is null) throw new ArgumentNullException(nameof(task));
        return Create(inputDim, hiddenNeurons, task.OutputDim(classCount), task.CreateOutputActivation(), seed);
    }
}
