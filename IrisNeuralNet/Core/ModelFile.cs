using IrisNeuralNet.Data;
using System;
using System.IO;

namespace IrisNeuralNet.Core
{
    public static class ModelFile
    {
        private const uint Magic = 0x4E4E4554u; // "NNET"
        private const int Version = 1;
        private const int MaxArrayLength = 1 << 24; // защита от раздувания аллокаций битым файлом

        public static void Save(NeuralNetwork network, FeatureNormalizer? normalizer, string path)
        {
            if (network is null) throw new ArgumentNullException(nameof(network));
            if (path is null) throw new ArgumentNullException(nameof(path));

            using var stream = new FileStream(path, FileMode.Create, FileAccess.Write);
            using var writer = new BinaryWriter(stream);

            writer.Write(Magic);
            writer.Write(Version);
            writer.Write(network.LayerCount);

            foreach (DenseLayer layer in network.Layers)
            {
                writer.Write(layer.InputDim);
                writer.Write(layer.OutputDim);
                writer.Write((byte)layer.ActivationKind);
                WriteFloats(writer, layer.Weights);
                WriteFloats(writer, layer.Biases);
            }

            WriteNormalizer(writer, normalizer);
        }

        public static LoadedModel Load(string path)
        {
            if (path is null) throw new ArgumentNullException(nameof(path));

            try
            {
                using var stream = new FileStream(path, FileMode.Open, FileAccess.Read);
                using var reader = new BinaryReader(stream);
                return ReadModel(reader);
            }
            catch (EndOfStreamException e)
            {
                throw new FormatException("Model file is truncated.", e);
            }
        }

        private static LoadedModel ReadModel(BinaryReader reader)
        {
            uint magic = reader.ReadUInt32();
            if (magic != Magic)
            {
                throw new FormatException($"Not a neural model file (bad magic 0x{magic:X8}).");
            }

            int version = reader.ReadInt32();
            if (version != Version)
            {
                throw new FormatException($"Unsupported model format version {version}.");
            }

            int layerCount = reader.ReadInt32();
            if (layerCount <= 0)
            {
                throw new FormatException($"Invalid layer count {layerCount}.");
            }

            var layers = new DenseLayer[layerCount];
            for (int i = 0; i < layerCount; i++)
            {
                layers[i] = ReadLayer(reader, i);
            }

            return new LoadedModel(new NeuralNetwork(layers), ReadNormalizer(reader));
        }

        private static DenseLayer ReadLayer(BinaryReader reader, int layerIndex)
        {
            int inputDim = reader.ReadInt32();
            int outputDim = reader.ReadInt32();
            if (inputDim <= 0 || outputDim <= 0)
            {
                throw new FormatException($"Layer {layerIndex}: invalid dimensions {inputDim}x{outputDim}.");
            }

            var kind = (ActivationKind)reader.ReadByte();
            float[] weights = ReadFloats(reader, inputDim * outputDim, $"layer {layerIndex} weights");
            float[] biases = ReadFloats(reader, outputDim, $"layer {layerIndex} biases");
            return new DenseLayer(weights, biases, ActivationFactory.Create(kind));
        }

        private static void WriteFloats(BinaryWriter writer, ReadOnlySpan<float> values)
        {
            writer.Write(values.Length);
            for (int i = 0; i < values.Length; i++)
            {
                writer.Write(values[i]);
            }
        }

        private static float[] ReadFloats(BinaryReader reader, int expectedLength, string what)
        {
            float[] values = ReadFloatArray(reader, what);
            if (values.Length != expectedLength)
            {
                throw new FormatException($"Corrupt model file: {what} count {values.Length}, expected {expectedLength}.");
            }

            return values;
        }

        private static float[] ReadFloatArray(BinaryReader reader, string what)
        {
            int count = reader.ReadInt32();
            if (count < 0 || count > MaxArrayLength)
            {
                throw new FormatException($"Corrupt model file: impossible {what} count {count}.");
            }

            var values = new float[count];
            for (int i = 0; i < count; i++)
            {
                values[i] = reader.ReadSingle();
            }

            return values;
        }

        private static void WriteNormalizer(BinaryWriter writer, FeatureNormalizer? normalizer)
        {
            if (normalizer is null)
            {
                writer.Write((byte)0);
                return;
            }

            writer.Write((byte)1);
            WriteFloats(writer, normalizer.Means);
            WriteFloats(writer, normalizer.Scales);
        }

        private static FeatureNormalizer? ReadNormalizer(BinaryReader reader)
        {
            byte flag = reader.ReadByte();
            if (flag == 0)
            {
                return null;
            }

            if (flag != 1)
            {
                throw new FormatException($"Corrupt model file: normalizer flag {flag}.");
            }

            float[] means = ReadFloatArray(reader, "normalizer means");
            float[] scales = ReadFloatArray(reader, "normalizer scales");
            if (means.Length == 0 || means.Length != scales.Length)
            {
                throw new FormatException("Corrupt model file: normalizer means/scales length mismatch.");
            }

            return FeatureNormalizer.FromParameters(means, scales);
        }
    }
}
