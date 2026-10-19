using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace IrisNeuralNet.Lab.Web;

/// <summary>Параметры запуска веб-лаборатории.</summary>
/// <param name="Port">0 — любой свободный порт.</param>
internal sealed record LabHostOptions(int Port, bool OpenBrowser)
{
    /// <summary>Не 5000: на macOS этот порт занят приёмником AirPlay.</summary>
    public const int DefaultPort = 5187;

    public static readonly string Usage =
        "Веб-лаборатория: IrisNeuralNet.Lab [--port <номер>] [--no-browser]\n" +
        $"  без аргументов   — http://127.0.0.1:{DefaultPort}, браузер откроется сам\n" +
        "  --port <номер>   — другой порт (0 — любой свободный)\n" +
        "  --no-browser     — не открывать браузер";

    public static bool TryParse(IReadOnlyList<string> args, out LabHostOptions options, out string error)
    {
        options = new LabHostOptions(DefaultPort, OpenBrowser: true);
        error = string.Empty;
        for (int i = 0; i < args.Count; i++)
        {
            switch (args[i])
            {
                case "--no-browser":
                    options = options with { OpenBrowser = false };
                    break;
                case "--port" when i + 1 < args.Count && int.TryParse(args[i + 1], out int port) && port is >= 0 and <= 65535:
                    options = options with { Port = port };
                    i++;
                    break;
                default:
                    error = $"Неизвестный аргумент: {args[i]}";
                    return false;
            }
        }

        return true;
    }
}

/// <summary>
/// Локальный веб-сервер лаборатории (Kestrel): слушает только 127.0.0.1 — управление обучением
/// не видно из сети. Статика страницы лежит в wwwroot рядом с exe, всё остальное — API.
/// </summary>
internal static class LabWebHost
{
    public static WebApplication Build(LabHostOptions options)
    {
        WebApplicationBuilder builder = WebApplication.CreateBuilder(new WebApplicationOptions
        {
            // Рядом с exe: и для dotnet run, и для распакованного релиза, откуда бы его ни запустили.
            ContentRootPath = AppContext.BaseDirectory,
        });

        builder.WebHost.UseUrls($"http://127.0.0.1:{options.Port}");
        builder.Logging.SetMinimumLevel(LogLevel.Warning);
        builder.Services.ConfigureHttpJsonOptions(json =>
        {
            json.SerializerOptions.Converters.Add(new JsonStringEnumConverter());
            json.SerializerOptions.NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals;
        });
        builder.Services.AddSingleton<LabState>();
        builder.Services.AddSingleton<SweepCoordinator>();

        WebApplication app = builder.Build();

        // Sweep подписывается на смену данных — создаём его сразу, а не при первом запросе.
        app.Services.GetRequiredService<SweepCoordinator>();

        app.UseDefaultFiles();
        app.UseStaticFiles(new StaticFileOptions
        {
            // После обновления версии браузер не должен показывать старый скрипт из кэша.
            OnPrepareResponse = context => context.Context.Response.Headers.CacheControl = "no-cache",
        });
        app.Use(async (context, next) =>
        {
            if (context.Request.Path.StartsWithSegments("/api")) context.Response.Headers.CacheControl = "no-store";
            await next(context);
        });
        LabEndpoints.Map(app);
        return app;
    }

    public static async Task<int> RunAsync(IReadOnlyList<string> args)
    {
        if (!LabHostOptions.TryParse(args, out LabHostOptions options, out string error))
        {
            Console.WriteLine(error);
            Console.WriteLine(LabHostOptions.Usage);
            return 2;
        }

        if (options.Port != 0 && !IsPortFree(options.Port))
        {
            Console.WriteLine($"Порт {options.Port} занят — беру любой свободный.");
            options = options with { Port = 0 };
        }

        await using WebApplication app = Build(options);
        await app.StartAsync();

        string url = app.Urls.First();
        Console.WriteLine($"Iris Neural Lab: {url}  (Ctrl+C — остановить)");
        if (options.OpenBrowser) OpenBrowser(url);

        await app.WaitForShutdownAsync();
        return 0;
    }

    private static bool IsPortFree(int port)
    {
        try
        {
            var listener = new TcpListener(IPAddress.Loopback, port);
            listener.Start();
            listener.Stop();
            return true;
        }
        catch (SocketException)
        {
            return false;
        }
    }

    private static void OpenBrowser(string url)
    {
        try
        {
            ProcessStartInfo start = OperatingSystem.IsWindows() ? new ProcessStartInfo(url) { UseShellExecute = true }
                : OperatingSystem.IsMacOS() ? new ProcessStartInfo("open", url)
                : new ProcessStartInfo("xdg-open", url);
            Process.Start(start)?.Dispose();
        }
        catch (Exception e) when (e is Win32Exception or InvalidOperationException)
        {
            Console.WriteLine("Не удалось открыть браузер — откройте адрес выше вручную.");
        }
    }
}
