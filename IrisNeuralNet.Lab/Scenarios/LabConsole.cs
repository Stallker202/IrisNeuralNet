using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using IrisNeuralNet.Lab.Reporting;

namespace IrisNeuralNet.Lab.Scenarios;

/// <summary>
/// Консольный режим: прогон практической части лабораторных с выводом в консоль,
/// results.md и PNG-графиками в папке отчёта.
/// </summary>
internal static class LabConsole
{
    public const string Usage =
        "Лабораторные в консоли: IrisNeuralNet.Lab --console [iris|sonar|boston|mnist|all ...] [--quick] [--out <папка>] [--image <файл>] [--no-charts]\n" +
        "  без датасета      — iris\n" +
        "  --quick           — в 10 раз меньше эпох (быстрая проверка)\n" +
        "  --out <папка>     — куда сохранить results.md и графики (по умолчанию ./reports)\n" +
        "  --image <файл>    — mnist: распознать своё изображение цифры (PNG)\n" +
        "  --no-charts       — не сохранять графики";

    private static readonly ILabScenario[] Scenarios =
    {
        new IrisScenario(),
        new SonarScenario(),
        new BostonScenario(),
        new MnistScenario(),
    };

    public static int Run(IReadOnlyList<string> args, IChartRenderer charts, IImageLoader? images)
    {
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

        if (!TryParse(args, charts, images, out List<ILabScenario> selected, out ScenarioOptions? options, out string outputDirectory, out string error))
        {
            Console.WriteLine(error);
            Console.WriteLine(Usage);
            return 2;
        }

        foreach (ILabScenario scenario in selected)
        {
            Console.WriteLine();
            Console.WriteLine(new string('#', 72));
            Console.WriteLine($"# {scenario.Title}");
            Console.WriteLine(new string('#', 72));

            var context = new ScenarioContext(scenario.Title, Path.Combine(outputDirectory, scenario.Key), options!, Console.Out);
            scenario.Run(context);
            Console.WriteLine();
            Console.WriteLine($"Отчёт: {context.Save()}");
        }

        return 0;
    }

    private static bool TryParse(
        IReadOnlyList<string> args, IChartRenderer charts, IImageLoader? images,
        out List<ILabScenario> selected, out ScenarioOptions? options, out string outputDirectory, out string error)
    {
        selected = new List<ILabScenario>();
        options = null;
        outputDirectory = Path.GetFullPath("reports");
        error = string.Empty;
        bool quick = false;
        string? imagePath = null;

        for (int i = 0; i < args.Count; i++)
        {
            string arg = args[i];
            switch (arg.ToLowerInvariant())
            {
                case "--quick":
                    quick = true;
                    break;
                case "--no-charts":
                    charts = NullChartRenderer.Instance;
                    break;
                case "--out" when i + 1 < args.Count:
                    outputDirectory = Path.GetFullPath(args[++i]);
                    break;
                case "--image" when i + 1 < args.Count:
                    imagePath = Path.GetFullPath(args[++i]);
                    break;
                case "all":
                    selected.AddRange(Scenarios);
                    break;
                default:
                    ILabScenario? scenario = Scenarios.FirstOrDefault(s => s.Key.Equals(arg, StringComparison.OrdinalIgnoreCase));
                    if (scenario is null)
                    {
                        error = $"Неизвестный аргумент: {arg}";
                        return false;
                    }

                    selected.Add(scenario);
                    break;
            }
        }

        if (selected.Count == 0) selected.Add(Scenarios[0]);
        selected = selected.Distinct().ToList();
        options = new ScenarioOptions(quick, charts, images, imagePath);
        return true;
    }
}
