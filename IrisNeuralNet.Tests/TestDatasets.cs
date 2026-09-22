using System;

namespace IrisNeuralNet.Tests
{
    internal static class TestDatasets
    {
        public static (float[] X, float[] Y) CreateTwoClusters(int rowsPerClass, int seed)
        {
            var random = new Random(seed);
            int rowCount = rowsPerClass * 2;
            var x = new float[rowCount * 2];
            var y = new float[rowCount * 2];

            int row = 0;
            for (int cls = 0; cls < 2; cls++)
            {
                for (int i = 0; i < rowsPerClass; i++)
                {
                    float center = cls == 0 ? 0.5f : 3.5f;
                    x[row * 2] = center + (float)(random.NextDouble() - 0.5) * 0.6f;
                    x[row * 2 + 1] = center + (float)(random.NextDouble() - 0.5) * 0.6f;
                    y[row * 2 + cls] = 1f;
                    row++;
                }
            }

            return (x, y);
        }
    }
}
