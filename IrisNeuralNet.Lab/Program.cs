using System;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using IrisNeuralNet.Lab.Scenarios;

namespace IrisNeuralNet.Lab;

internal static class Program
{
    /// <summary>
    /// Без аргументов — окно лаборатории; <c>--console [iris|sonar|boston|mnist|all] [--quick] [--out папка] [--image файл]</c> —
    /// практическая часть лабораторных в консоли с results.md и PNG-графиками.
    /// </summary>
    [STAThread]
    private static int Main(string[] args)
    {
        try
        {
            if (args.Length > 0 && args[0] is "--help" or "-h" or "/?")
            {
                Console.WriteLine(LabConsole.Usage);
                return 0;
            }

            if (args.Length > 0 && args[0] == "--console")
            {
                Console.OutputEncoding = Encoding.UTF8;
                return LabConsole.Run(args.Skip(1).ToArray(), new PngChartRenderer(), new BitmapImageLoader());
            }

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new LabForm());
            return 0;
        }
        catch (Exception e)
        {
            Console.WriteLine(e);
            return 1;
        }
    }
}
