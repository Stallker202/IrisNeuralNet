using System.Windows.Forms;

namespace IrisNeuralNet.Lab;

/// <summary>PictureBox без мерцания: двойная буферизация для живой отрисовки графиков.</summary>
internal sealed class ChartBox : PictureBox
{
    public ChartBox()
    {
        SetStyle(
            ControlStyles.OptimizedDoubleBuffer | ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint,
            true);
        UpdateStyles();
    }
}