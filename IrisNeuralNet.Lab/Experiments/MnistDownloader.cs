using System;
using System.IO;
using System.Net.Http;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;
using IrisNeuralNet.Data;

namespace IrisNeuralNet.Lab.Experiments;

/// <summary>
/// Скачивает MNIST при первом обращении — как keras.datasets.mnist.load_data() кладёт файл в ~/.keras/datasets.
/// Файлы проверяются по SHA-256 оригинальных архивов LeCun, поэтому любое зеркало даёт те же байты.
/// Офлайн-вариант: положить четыре файла в папку «mnist» рядом с exe.
/// </summary>
internal static class MnistDownloader
{
    private static readonly string[] Mirrors =
    {
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "https://raw.githubusercontent.com/fgnt/mnist/master/",
    };

    private static readonly (string FileName, string Sha256)[] Files =
    {
        (MnistDataLoader.TrainImagesFile, "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"),
        (MnistDataLoader.TrainLabelsFile, "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"),
        (MnistDataLoader.TestImagesFile, "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6"),
        (MnistDataLoader.TestLabelsFile, "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6"),
    };

    /// <summary>Кэш как у Keras: %LOCALAPPDATA%/IrisNeuralNet/datasets/mnist.</summary>
    public static string CacheDirectory =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "IrisNeuralNet", "datasets", "mnist");

    /// <summary>Папка «mnist» рядом с exe (офлайн-копия) или кэш; null, если файлов нет нигде.</summary>
    public static string? FindLocalDirectory()
    {
        string bundled = DataFiles.Resolve("mnist");
        if (MnistDataLoader.IsAvailable(bundled)) return bundled;
        return MnistDataLoader.IsAvailable(CacheDirectory) ? CacheDirectory : null;
    }

    public static async Task<string> EnsureAsync(IProgress<string>? progress, CancellationToken token)
    {
        string? existing = FindLocalDirectory();
        if (existing is not null) return existing;

        Directory.CreateDirectory(CacheDirectory);
        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(5) };
        foreach ((string fileName, string sha256) in Files)
        {
            string target = Path.Combine(CacheDirectory, fileName);
            if (File.Exists(target) && await HashMatchesAsync(target, sha256, token).ConfigureAwait(false))
            {
                continue;
            }

            await DownloadFromAnyMirrorAsync(http, fileName, sha256, target, progress, token).ConfigureAwait(false);
        }

        progress?.Report($"MNIST готов: {CacheDirectory}");
        return CacheDirectory;
    }

    private static async Task DownloadFromAnyMirrorAsync(
        HttpClient http, string fileName, string sha256, string target, IProgress<string>? progress, CancellationToken token)
    {
        Exception? lastError = null;
        foreach (string mirror in Mirrors)
        {
            string temporary = target + ".part";
            try
            {
                progress?.Report($"скачиваю {fileName} ({new Uri(mirror).Host})…");
                using (HttpResponseMessage response = await http.GetAsync(mirror + fileName, HttpCompletionOption.ResponseHeadersRead, token).ConfigureAwait(false))
                {
                    response.EnsureSuccessStatusCode();
                    await using FileStream file = File.Create(temporary);
                    await response.Content.CopyToAsync(file, token).ConfigureAwait(false);
                }

                if (!await HashMatchesAsync(temporary, sha256, token).ConfigureAwait(false))
                {
                    throw new InvalidDataException($"SHA-256 mismatch for {fileName} from {mirror}.");
                }

                File.Move(temporary, target, overwrite: true);
                return;
            }
            catch (Exception e) when (e is HttpRequestException or InvalidDataException or IOException or TaskCanceledException && !token.IsCancellationRequested)
            {
                lastError = e;
                File.Delete(temporary);
            }
        }

        throw new IOException($"Не удалось скачать {fileName} ни с одного зеркала. Положите файлы MNIST в папку 'mnist' рядом с exe.", lastError);
    }

    private static async Task<bool> HashMatchesAsync(string path, string expectedSha256, CancellationToken token)
    {
        await using FileStream file = File.OpenRead(path);
        byte[] hash = await SHA256.HashDataAsync(file, token).ConfigureAwait(false);
        return Convert.ToHexString(hash).Equals(expectedSha256, StringComparison.OrdinalIgnoreCase);
    }
}
