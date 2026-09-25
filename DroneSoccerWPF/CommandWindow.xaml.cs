using System.Windows;

namespace DroneSoccerWPF;

public partial class CommandWindow : Window
{
    private readonly object _gate = new();
    private int _lines;
    private bool _allowClose;

    public CommandWindow()
    {
        InitializeComponent();
        WriteLine("Command log ready. Connect the drone to see the commands sent to it.");
    }

    public void AllowClose() => _allowClose = true;

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

    protected override void OnClosing(System.ComponentModel.CancelEventArgs e)
    {
        if (_allowClose)
        {
            base.OnClosing(e);
            return;
        }

        e.Cancel = true;
        Hide();
    }
}
