using System.Windows.Controls;
using System.Windows.Media;

namespace DroneSoccerWPF;

public partial class ExecutionPanel : UserControl
{
    private static readonly SolidColorBrush BrushGreen = Freeze(0x3D, 0xDC, 0x97);
    private static readonly SolidColorBrush BrushAmber = Freeze(0xFF, 0xB0, 0x20);
    private static readonly SolidColorBrush BrushRed = Freeze(0xFF, 0x5A, 0x6A);
    private static readonly SolidColorBrush BrushMuted = Freeze(0xC5, 0xD0, 0xC2);

    private readonly object _gate = new();
    private string _lastPhase = "";
    private int _lines;

    public ExecutionPanel()
    {
        InitializeComponent();
        WriteLine("Execution log ready. Connect steps and current status print here.");
    }

    public string Transcript => ConsoleBox.Text;

    public void WriteLine(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return;

        var line = $"{DateTime.Now:HH:mm:ss}  {text}";
        lock (_gate)
        {
            ConsoleBox.AppendText(line + Environment.NewLine);
            _lines++;
            if (_lines > 400)
            {
                var content = ConsoleBox.Text;
                var cut = content.IndexOf('\n');
                if (cut >= 0)
                {
                    ConsoleBox.Text = content[(cut + 1)..];
                    _lines--;
                }
            }

            ConsoleBox.CaretIndex = ConsoleBox.Text.Length;
            ConsoleBox.ScrollToEnd();
        }
    }

    public void SetStatus(string phase, string detail, bool repeat)
    {
        StatusPhase.Text = phase;
        StatusDetail.Text = detail;
        var color = phase switch
        {
            "CONNECTED" => BrushGreen,
            "CONNECTING" => BrushAmber,
            "FAILED" => BrushRed,
            _ => BrushMuted,
        };
        StatusPhase.Foreground = color;
        StatusCard.BorderBrush = color;

        var changed = phase != _lastPhase;
        _lastPhase = phase;
        if (!changed && !(repeat && phase is "CONNECTED" or "CONNECTING"))
            return;

        WriteLine($"STATUS  {phase}  {detail}");
    }

    private static SolidColorBrush Freeze(byte red, byte green, byte blue)
    {
        var brush = new SolidColorBrush(Color.FromRgb(red, green, blue));
        brush.Freeze();
        return brush;
    }
}
