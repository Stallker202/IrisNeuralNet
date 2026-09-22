using System;
using System.Collections.Generic;
using IrisNeuralNet.Diagnostics;

namespace IrisNeuralNet.Lab
{
    internal sealed class ConsoleTrainingObserver : ITrainingObserver
    {
        private readonly List<float> _trainLoss = new();
        private readonly List<float> _trainAccuracy = new();
        private readonly List<float> _validationLoss = new();
        private readonly List<float> _validationAccuracy = new();

        public void OnEpochCompleted(in EpochMetrics metrics)
        {
            _trainLoss.Add(metrics.TrainLoss);
            _trainAccuracy.Add(metrics.TrainAccuracy);
            _validationLoss.Add(metrics.ValidationLoss);
            _validationAccuracy.Add(metrics.ValidationAccuracy);

            if (metrics.Epoch == 1 || metrics.Epoch % 10 == 0)
            {
                Console.WriteLine(
                    $"epoch {metrics.Epoch,4} | loss {metrics.TrainLoss:F4} | acc {metrics.TrainAccuracy,6:P1} | val_loss {metrics.ValidationLoss:F4} | val_acc {metrics.ValidationAccuracy,6:P1}");
            }
        }

        public void PrintCharts()
        {
            AsciiChart.Print("Loss", _trainLoss, _validationLoss, "train", "validation");
            AsciiChart.Print("Accuracy", _trainAccuracy, _validationAccuracy, "train", "validation");
        }
    }
}
