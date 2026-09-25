using BenchmarkDotNet.Attributes;
using IrisNeuralNet.MathCore;
using System;

namespace IrisNeuralNet.Benchmarks
{
    [MemoryDiagnoser]
    public class MathBenchmarks
    {
        private const int Rows = 128;
        private const int Inner = 128;
        private const int Cols = 128;

        private float[] _left = Array.Empty<float>();
        private float[] _right = Array.Empty<float>();
        private float[] _dest = Array.Empty<float>();

        [GlobalSetup]
        public void Setup()
        {
            var random = new Random(7);
            _left = Fill(random, Rows * Inner);
            _right = Fill(random, Inner * Cols);
            _dest = new float[Rows * Cols];
        }

        [Benchmark(Baseline = true)]
        public float NaiveTripleLoop()
        {
            NaiveMatMul(_left, _right, _dest, Rows, Inner, Cols);
            return _dest[0];
        }

        [Benchmark]
        public float SimdMatMul()
        {
            VectorOps.MatMul(_left, _right, _dest, Rows, Inner, Cols);
            return _dest[0];
        }

        [Benchmark]
        public float SimdDot() => VectorOps.Dot(_left, _right);

        private static float[] Fill(Random random, int length)
        {
            var data = new float[length];
            for (int i = 0; i < length; i++)
            {
                data[i] = (float)(random.NextDouble() * 2 - 1);
            }

            return data;
        }

        private static void NaiveMatMul(float[] left, float[] right, float[] dest, int rows, int inner, int cols)
        {
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float sum = 0f;
                    for (int k = 0; k < inner; k++)
                    {
                        sum += left[i * inner + k] * right[k * cols + j];
                    }

                    dest[i * cols + j] = sum;
                }
            }
        }
    }
}
