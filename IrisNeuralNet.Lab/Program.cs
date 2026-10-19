using System;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using IrisNeuralNet.Lab.Imaging;
using IrisNeuralNet.Lab.Rendering;
using IrisNeuralNet.Lab.Scenarios;
using IrisNeuralNet.Lab.Web;

namespace IrisNeuralNet.Lab;

internal static class Program
{
    /// <summary>
    /// Без аргументов — веб-лаборатория в браузере (Windows, macOS, Linux);
    /// <c>--console …</c> — практическая часть лабораторных с results.md и SVG-графиками.
    /// </summary>
    private static async Task<int> Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;
        try
        {
            if (args.Length > 0 && args[0] is "--help" or "-h" or "/?")
            {
                Console.WriteLine(LabHostOptions.Usage);
                Console.WriteLine();
                Console.WriteLine(LabConsole.Usage);
                return 0;
            }

            if (args.Length > 0 && args[0] == "--console")
            {
                return LabConsole.Run(args.Skip(1).ToArray(), new SvgChartRenderer(), new PngImageLoader());
            }

            return await LabWebHost.RunAsync(args);
        }
        catch (Exception e)
        {
            Console.WriteLine(e);
            return 1;
        }
    }
}
