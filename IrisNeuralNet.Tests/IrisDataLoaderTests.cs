using System;
using System.IO;
using IrisNeuralNet.Data;
using Xunit;

namespace IrisNeuralNet.Tests
{
    public sealed class IrisDataLoaderTests
    {
        [Fact]
        public void Load_ParsesFeaturesClassesAndBlankLines()
        {
            string path = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".csv");
            File.WriteAllText(path,
                "5.1,3.5,1.4,0.2,Iris-setosa\r\n" +
                "\r\n" +
                "6.7,3.0,5.2,2.3,Iris-virginica\n" +
                "5.9,3.0,4.2,1.5,Iris-versicolor"); // без завершающего перевода строки
            try
            {
                IrisDataset dataset = IrisDataLoader.Load(path);

                Assert.Equal(3, dataset.RowCount);
                Assert.Equal(new float[] { 5.1f, 3.5f, 1.4f, 0.2f, 6.7f, 3.0f, 5.2f, 2.3f, 5.9f, 3.0f, 4.2f, 1.5f },
                    dataset.Features.ToArray());
                Assert.Equal(new int[] { 0, 2, 1 }, dataset.Classes.ToArray());
            }
            finally
            {
                File.Delete(path);
            }
        }

        [Fact]
        public void Load_Throws_OnUnknownClassLabel()
        {
            string path = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".csv");
            File.WriteAllText(path, "5.1,3.5,1.4,0.2,Iris-unknown\n");
            try
            {
                Assert.Throws<FormatException>(() => IrisDataLoader.Load(path));
            }
            finally
            {
                File.Delete(path);
            }
        }

        [Fact]
        public void Load_Throws_OnMalformedRow()
        {
            string path = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".csv");
            File.WriteAllText(path, "5.1,3.5,1.4,Iris-setosa\n");
            try
            {
                Assert.Throws<FormatException>(() => IrisDataLoader.Load(path));
            }
            finally
            {
                File.Delete(path);
            }
        }
    }
}
