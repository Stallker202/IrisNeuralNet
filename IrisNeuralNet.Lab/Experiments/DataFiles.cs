using System;
using System.IO;

namespace IrisNeuralNet.Lab.Experiments;

/// <summary>
/// Поиск файлов данных: сначала текущая папка (запуск из консоли рядом с csv),
/// затем папка exe (запуск из Visual Studio, где рабочая папка — bin).
/// </summary>
internal static class DataFiles
{
    public static string Resolve(string fileName)
    {
        string local = Path.GetFullPath(fileName);
        if (File.Exists(local) || Directory.Exists(local)) return local;

        return Path.Combine(AppContext.BaseDirectory, fileName);
    }

    public static bool Exists(string fileName) => File.Exists(Resolve(fileName));
}
