using BenchmarkDotNet.Attributes;
using IrisNeuralNet.Core;
using IrisNeuralNet.Optimizers;
using System;

namespace IrisNeuralNet.Benchmarks
{
    [MemoryDiagnoser]
    [IterationCount(10)]
    [WarmupCount(3)]
    public class TrainingBenchmarks
    {
        private float[] _x = Array.Empty<float>();
        private float[] _y = Array.Empty<float>();
        private Trainer _trainer = null!;

        [GlobalSetup]
        [IterationSetup]
        public void Setup()
        {
            var random = new Random(11);
            const int rowsPerClass = 60;
            int rowCount = rowsPerClass * 3;
            _x = new float[rowCount * 4];
            _y = new float[rowCount * 3];

            int row = 0;
            for (int cls = 0; cls < 3; cls++)
            {
                for (int i = 0; i < rowsPerClass; i++)
                {
                    float center = cls * 4f;
                    for (int f = 0; f < 4; f++)
                    {
                        _x[row * 4 + f] = center + (float)(random.NextDouble() - 0.5);
                    }

                    _y[row * 3 + cls] = 1f;
                    row++;
                }
            }

            // Пересоздаём сеть на каждую итерацию: измеряем всегда одинаковое состояние.
            NeuralNetwork network = NetworkFactory.Create(4, new[] { 16, 16 }, 3, seed: 5);
            _trainer = new Trainer(network, new AdamOptimizer(network, 0.01f), new CategoricalCrossEntropyLoss());
        }

        [Benchmark]
        public float FitTwentyEpochs() =>
            _trainer.Fit(_x, _y, epochs: 20, batchSize: 32, seed: 3);
    }
}
