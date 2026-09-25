using System;
using System.Windows.Forms;
using IrisNeuralNet.Core;
using IrisNeuralNet.Data;
using IrisNeuralNet.Optimizers;

namespace IrisNeuralNet.Lab;

internal static class Program
{
    private const string DatasetFileName = "iris.csv";
    private const string ModelFileName = "iris_model.nn";
    private const float ValidationFraction = 0.2f;
    private const int SplitSeed = 42;
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
            Console.WriteLine($"Ошибка: {e.Message}");
            return 1;
        }
    }

    private static void RunUi()
    {
        PreparedData data = PrepareData();
        using var session = new TrainingSession(
            data.TrainFeatures, data.TrainLabels,
            data.ValidationFeatures, data.ValidationLabels,
            IrisDataLoader.FeatureCount, data.ClassCount);

        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        Application.Run(new LabForm(session));
    }

    private static void RunConsole()
    {
        PreparedData data = PrepareData();
        NeuralNetwork network = BuildNetwork(data.ClassCount);
        ConsoleTrainingObserver observer = new ConsoleTrainingObserver();
        Trainer trainer = new Trainer(
            network,
            new AdamOptimizer(network, LearningRate),
            new CategoricalCrossEntropyLoss(),
            observer);

        Console.WriteLine($"строк: {data.RowCount} (train {data.TrainRowCount} / val {data.ValidationRowCount})");
        trainer.Fit(data.TrainFeatures, data.TrainLabels, data.ValidationFeatures, data.ValidationLabels,
            Epochs, BatchSize, TrainingSeed);

        Console.WriteLine();
        Console.WriteLine($"train accuracy: {trainer.EvaluateAccuracy(data.TrainFeatures, data.TrainLabels):P1}");
        Console.WriteLine($"val   accuracy: {trainer.EvaluateAccuracy(data.ValidationFeatures, data.ValidationLabels):P1}");
        observer.PrintCharts();
        SaveAndVerifyModel(network, data.Normalizer, data);
    }

    private static PreparedData PrepareData()
    {
        IrisDataset dataset = IrisDataLoader.Load(DatasetFileName);
        (IrisDataset train, IrisDataset validation) = DatasetSplitter.Split(dataset, ValidationFraction, SplitSeed);

        float[] trainFeatures = train.Features.ToArray();
        float[] validationFeatures = validation.Features.ToArray();

        FeatureNormalizer normalizer = FeatureNormalizer.Fit(trainFeatures, IrisDataLoader.FeatureCount, train.RowCount);
        normalizer.Apply(trainFeatures, train.RowCount);
        normalizer.Apply(validationFeatures, validation.RowCount);

        return new PreparedData
        {
            ClassCount = dataset.ClassCount,
            RowCount = dataset.RowCount,
            TrainRowCount = train.RowCount,
            ValidationRowCount = validation.RowCount,
            TrainFeatures = trainFeatures,
            TrainLabels = OneHotEncoder.Encode(train.Classes, dataset.ClassCount),
            ValidationFeatures = validationFeatures,
            ValidationLabels = OneHotEncoder.Encode(validation.Classes, dataset.ClassCount),
            RawValidationFeatures = validation.Features.ToArray(),
            ValidationClasses = validation.Classes.ToArray(),
            Normalizer = normalizer,
        };
    }

    private static NeuralNetwork BuildNetwork(int classCount) =>
        new NeuralNetwork(new[]
        {
            new DenseLayer(IrisDataLoader.FeatureCount, HiddenNeurons, new ReLUActivation(), seed: 1),
            new DenseLayer(HiddenNeurons, classCount, new SoftmaxActivation(), seed: 2),
        });

    private static void SaveAndVerifyModel(NeuralNetwork network, FeatureNormalizer normalizer, PreparedData data)
    {
        ModelFile.Save(network, normalizer, ModelFileName);
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
                rawFeatures.AsSpan(i * IrisDataLoader.FeatureCount, IrisDataLoader.FeatureCount));
            if (predicted == data.ValidationClasses[i])
            {
                correct++;
            }
        }

        Console.WriteLine($"loaded model val accuracy: {(float)correct / data.ValidationRowCount:P1}");
    }
}

internal sealed class PreparedData
{
    public required int ClassCount { get; init; }
    public required int RowCount { get; init; }
    public required int TrainRowCount { get; init; }
    public required int ValidationRowCount { get; init; }
    public required float[] TrainFeatures { get; init; }
    public required float[] TrainLabels { get; init; }
    public required float[] ValidationFeatures { get; init; }
    public required float[] ValidationLabels { get; init; }
    public required float[] RawValidationFeatures { get; init; }
    public required int[] ValidationClasses { get; init; }
    public required FeatureNormalizer Normalizer { get; init; }
}