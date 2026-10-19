using System;
using System.Windows.Forms;
using IrisNeuralNet.Core;
using IrisNeuralNet.Data;
using IrisNeuralNet.Optimizers;

namespace IrisNeuralNet.Lab
{

    internal static class Program
    {
        private const string ModelFileName = "iris_model.nn";
        private const int HiddenNeurons = 8;
        private const int Epochs = 120;
        private const int BatchSize = 16;
        private const int TrainingSeed = 7;
        private const float LearningRate = 0.02f;

        [STAThread]
        private static int Main(string[] args)
        {
            try
            {
                if (args.Length > 0 && args[0] == "--console")
                {
                    RunConsole();
                }
                else
                {
                    RunUi();
                }

                return 0;
            }
            catch (Exception e)
            {
                Console.WriteLine(e.ToString()); 
                return 1;
            }
        }

        private static void RunUi()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new LabForm());
        }

        private static void RunConsole()
        {
            PreparedData data = DataPreparation.Prepare(Datasets.Iris, 0.2f, normalize: true);

            NeuralNetwork network = NetworkFactory.Create(IrisDataLoader.FeatureCount, new[] { HiddenNeurons }, data.ClassCount, seed: 1);
            ConsoleTrainingObserver observer = new ConsoleTrainingObserver();
            Trainer trainer = new Trainer(
                network,
                new AdamOptimizer(network, LearningRate),
                new CategoricalCrossEntropyLoss(),
                observer);

            Console.WriteLine($"Iris: {data.RowCount} образцов (train {data.TrainRowCount} / val {data.ValidationRowCount})");
            trainer.Fit(data.TrainFeatures, data.TrainLabels, data.ValidationFeatures, data.ValidationLabels,
                Epochs, BatchSize, TrainingSeed);

            Console.WriteLine();
            Console.WriteLine($"train accuracy: {trainer.EvaluateAccuracy(data.TrainFeatures, data.TrainLabels):P1}");
            Console.WriteLine($"val   accuracy: {trainer.EvaluateAccuracy(data.ValidationFeatures, data.ValidationLabels):P1}");
            observer.PrintCharts();
            SaveAndVerifyModel(network, data);
        }

        private static void SaveAndVerifyModel(NeuralNetwork network, PreparedData data)
        {
            ModelFile.Save(network, data.Normalizer, ModelFileName);
            LoadedModel loaded = ModelFile.Load(ModelFileName);
            Console.WriteLine();
            Console.WriteLine($"model saved to '{ModelFileName}' and reloaded ({network.LayerCount} layers)");

            float[] rawFeatures = (float[])data.RawValidationFeatures.Clone();
            if (loaded.Normalizer is not null)
            {
                loaded.Normalizer.Apply(rawFeatures, data.ValidationRowCount);
            }

            int correct = 0;
            for (int i = 0; i < data.ValidationRowCount; i++)
            {
                int predicted = loaded.Network.PredictClass(
                    rawFeatures.AsSpan(i * data.FeatureCount, data.FeatureCount));
                if (predicted == data.ValidationClasses[i])
                {
                    correct++;
                }
            }

            Console.WriteLine($"loaded model val accuracy: {(float)correct / data.ValidationRowCount:P1}");
        }
    }
}