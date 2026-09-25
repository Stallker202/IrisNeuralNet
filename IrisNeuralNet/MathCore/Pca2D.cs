using System;

namespace IrisNeuralNet.MathCore
{
    public sealed class Pca2D
    {
        private readonly float[] _mean;
        private readonly float[] _components; // [featureIndex * 2 + componentIndex]
        private readonly float _p1Min, _p1Max, _p2Min, _p2Max;

        public float P1Min => _p1Min;
        public float P1Max => _p1Max;
        public float P2Min => _p2Min;
        public float P2Max => _p2Max;

        private Pca2D(float[] mean, float[] components, float p1Min, float p1Max, float p2Min, float p2Max)
        {
            _mean = mean;
            _components = components;
            _p1Min = p1Min;
            _p1Max = p1Max;
            _p2Min = p2Min;
            _p2Max = p2Max;
        }

        public static Pca2D Fit(ReadOnlySpan<float> features, int dim)
        {
            if (dim < 2) throw new ArgumentException("PCA requires at least 2 dimensions.", nameof(dim));
            int rows = features.Length / dim;
            if (rows < 2) throw new ArgumentException("Not enough rows for PCA.", nameof(features));

            var mean = new float[dim];
            for (int r = 0; r < rows; r++)
            {
                for (int i = 0; i < dim; i++)
                {
                    mean[i] += features[r * dim + i];
                }
            }

            for (int i = 0; i < dim; i++)
            {
                mean[i] /= rows;
            }

            var cov = new float[dim, dim];
            for (int r = 0; r < rows; r++)
            {
                for (int i = 0; i < dim; i++)
                {
                    float di = features[r * dim + i] - mean[i];
                    for (int j = i; j < dim; j++)
                    {
                        cov[i, j] += di * (features[r * dim + j] - mean[j]);
                    }
                }
            }

            for (int i = 0; i < dim; i++)
            {
                for (int j = i; j < dim; j++)
                {
                    cov[i, j] /= rows;
                    cov[j, i] = cov[i, j];
                }
            }

            var vectors = new float[dim, dim];
            JacobiEigen(cov, vectors, dim);

            int[] order = new int[dim];
            for (int i = 0; i < dim; i++)
            {
                order[i] = i;
            }

            Array.Sort(order, (a, b) => cov[b, b].CompareTo(cov[a, a]));

            var components = new float[dim * 2];
            for (int i = 0; i < dim; i++)
            {
                components[i * 2] = vectors[i, order[0]];
                components[i * 2 + 1] = vectors[i, order[1]];
            }

            float p1Min = float.MaxValue, p1Max = float.MinValue;
            float p2Min = float.MaxValue, p2Max = float.MinValue;
            for (int r = 0; r < rows; r++)
            {
                ProjectRow(features.Slice(r * dim, dim), mean, components, dim, out float p1, out float p2);
                p1Min = MathF.Min(p1Min, p1); p1Max = MathF.Max(p1Max, p1);
                p2Min = MathF.Min(p2Min, p2); p2Max = MathF.Max(p2Max, p2);
            }

            float pad1 = (p1Max - p1Min) * 0.08f + 1e-6f;
            float pad2 = (p2Max - p2Min) * 0.08f + 1e-6f;
            return new Pca2D(mean, components, p1Min - pad1, p1Max + pad1, p2Min - pad2, p2Max + pad2);
        }

        public float Component(int featureIndex, int component) => _components[featureIndex * 2 + component];

        public void Project(ReadOnlySpan<float> sample, out float p1, out float p2) =>
            ProjectRow(sample, _mean, _components, _mean.Length, out p1, out p2);

        public void Reconstruct(float p1, float p2, Span<float> destination)
        {
            for (int i = 0; i < _mean.Length; i++)
            {
                destination[i] = _mean[i] + _components[i * 2] * p1 + _components[i * 2 + 1] * p2;
            }
        }

        private static void ProjectRow(
            ReadOnlySpan<float> sample, float[] mean, float[] components, int dim,
            out float p1, out float p2)
        {
            p1 = 0f;
            p2 = 0f;
            for (int i = 0; i < dim; i++)
            {
                float d = sample[i] - mean[i];
                p1 += d * components[i * 2];
                p2 += d * components[i * 2 + 1];
            }
        }

        private static void JacobiEigen(float[,] a, float[,] v, int n)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    v[i, j] = i == j ? 1f : 0f;
                }
            }

            for (int sweep = 0; sweep < 100; sweep++)
            {
                float off = 0f;
                for (int i = 0; i < n; i++)
                {
                    for (int j = i + 1; j < n; j++)
                    {
                        off += a[i, j] * a[i, j];
                    }
                }

                if (off < 1e-12f)
                {
                    return;
                }

                for (int p = 0; p < n; p++)
                {
                    for (int q = p + 1; q < n; q++)
                    {
                        if (MathF.Abs(a[p, q]) < 1e-12f)
                        {
                            continue;
                        }

                        float theta = (a[q, q] - a[p, p]) / (2f * a[p, q]);
                        float sign = theta >= 0f ? 1f : -1f;
                        float t = sign / (MathF.Abs(theta) + MathF.Sqrt(theta * theta + 1f));
                        float c = 1f / MathF.Sqrt(t * t + 1f);
                        float s = t * c;

                        for (int k = 0; k < n; k++)
                        {
                            float kp = a[k, p], kq = a[k, q];
                            a[k, p] = c * kp - s * kq;
                            a[k, q] = s * kp + c * kq;
                        }

                        for (int k = 0; k < n; k++)
                        {
                            float pk = a[p, k], qk = a[q, k];
                            a[p, k] = c * pk - s * qk;
                            a[q, k] = s * pk + c * qk;
                        }

                        for (int k = 0; k < n; k++)
                        {
                            float kp = v[k, p], kq = v[k, q];
                            v[k, p] = c * kp - s * kq;
                            v[k, q] = s * kp + c * kq;
                        }
                    }
                }
            }
        }
    }
}
