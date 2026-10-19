using System;
using System.Globalization;
using System.Text;
using IrisNeuralNet.Core;

namespace IrisNeuralNet.Diagnostics;

/// <summary>Текстовая сводка архитектуры в стиле Keras model.summary().</summary>
public static class ModelSummary
{
    private const int NameWidth = 30;
    private const int ShapeWidth = 20;
    private const int LineWidth = 65;

    /// <param name="inputShape">
    /// Исходная форма образца, если она многомерна (например, 28×28 для MNIST):
    /// тогда первой строкой выводится слой Flatten без параметров.
    /// </param>
    public static string Format(NeuralNetwork network, ReadOnlySpan<int> inputShape = default)
    {
        if (network is null) throw new ArgumentNullException(nameof(network));

        var text = new StringBuilder();
        string rule = new('_', LineWidth);
        text.AppendLine(rule);
        AppendRow(text, "Layer (type)", "Output Shape", "Param #");
        text.AppendLine(new string('=', LineWidth));

        if (inputShape.Length > 1)
        {
            AppendRow(text, "flatten (Flatten)", Shape(network.InputDim), "0");
        }

        for (int i = 0; i < network.LayerCount; i++)
        {
            DenseLayer layer = network.Layers[i];
            string name = i == 0 ? "dense" : $"dense_{i}";
            string activation = layer.ActivationKind.ToString().ToLowerInvariant();
            AppendRow(text, $"{name} (Dense, {activation})", Shape(layer.OutputDim), Number(layer.ParameterCount));
        }

        text.AppendLine(new string('=', LineWidth));
        text.AppendLine($"Total params: {Number(network.ParameterCount)}");
        text.AppendLine($"Trainable params: {Number(network.ParameterCount)}");
        text.AppendLine("Non-trainable params: 0");
        text.Append(rule);
        return text.ToString();
    }

    /// <summary>Компактная запись архитектуры: «4 → 8 → 3».</summary>
    public static string Architecture(NeuralNetwork network)
    {
        if (network is null) throw new ArgumentNullException(nameof(network));

        var text = new StringBuilder().Append(network.InputDim);
        foreach (DenseLayer layer in network.Layers)
        {
            text.Append(" → ").Append(layer.OutputDim);
        }

        return text.ToString();
    }

    private static void AppendRow(StringBuilder text, string name, string shape, string parameters) =>
        text.Append(' ').Append(name.PadRight(NameWidth)).Append(shape.PadRight(ShapeWidth)).AppendLine(parameters);

    private static string Shape(int width) => $"(None, {width})";

    private static string Number(int value) => value.ToString("N0", CultureInfo.InvariantCulture);
}
