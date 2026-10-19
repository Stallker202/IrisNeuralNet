using System.Collections.Generic;

namespace IrisNeuralNet.Lab.Reporting;

/// <summary>Именованный ряд значений по эпохам — независим от библиотеки рисования.</summary>
internal sealed record PlotSeries(string Name, IReadOnlyList<float> Values);

/// <summary>Изображение цифры 28×28 (пиксели 0..255) с подписью — для сеток образцов MNIST.</summary>
internal sealed record DigitSample(float[] Pixels, string Caption);
