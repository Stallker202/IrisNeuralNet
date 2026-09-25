using System;
using Xunit;
using IrisNeuralNet.MathCore;

namespace IrisNeuralNet.Tests
{
    public sealed class Pca2DTests
    {
        [Fact]
        public void FirstComponentAlignsWithDominantVariance()
        {
            var features = new float[30 * 2];
            for (int i = 0; i < 30; i++)
            {
                float t = i - 14.5f;
                features[i * 2] = t;
                features[i * 2 + 1] = 0.05f * t;
            }

            Pca2D pca = Pca2D.Fit(features, dim: 2);

            Assert.True(MathF.Abs(pca.Component(0, 0)) > 0.9f);
        }

        [Fact]
        public void ProjectReconstruct_RoundTripsOnRankTwoData()
        {
            float ux = 0.6f, uy = 0.8f;
            var features = new float[13 * 9 * 3];
            int row = 0;
            for (int a = -6; a <= 6; a++)
            {
                for (int b = -4; b <= 4; b++)
                {
                    features[row * 3] = a * ux;
                    features[row * 3 + 1] = a * uy;
                    features[row * 3 + 2] = b;
                    row++;
                }
            }

            Pca2D pca = Pca2D.Fit(features, dim: 3);

            for (int i = 0; i < row; i += 17)
            {
                ReadOnlySpan<float> sample = features.AsSpan(i * 3, 3);
                pca.Project(sample, out float p1, out float p2);
                float[] reconstructed = new float[3];
                pca.Reconstruct(p1, p2, reconstructed);
                for (int k = 0; k < 3; k++)
                {
                    Assert.True(MathF.Abs(reconstructed[k] - sample[k]) < 1e-2f,
                        $"dim {k}: {reconstructed[k]} vs {sample[k]}");
                }
            }
        }
    }
}
