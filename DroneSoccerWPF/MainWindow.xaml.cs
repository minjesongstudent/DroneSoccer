using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

namespace DroneSoccerWPF;

public partial class MainWindow : Window
{
    private static readonly SolidColorBrush BrushMuted = MakeBrush(0x9A, 0xA3, 0xB2);
    private static readonly SolidColorBrush BrushText = MakeBrush(0xF2, 0xF4, 0xF8);
    private static readonly SolidColorBrush BrushGreen = MakeBrush(0x3D, 0xDC, 0x97);
    private static readonly SolidColorBrush BrushAmber = MakeBrush(0xFF, 0xB0, 0x20);
    private static readonly SolidColorBrush BrushRed = MakeBrush(0xFF, 0x5A, 0x6A);
    private static readonly SolidColorBrush BrushAccent = MakeBrush(0x7E, 0xB6, 0xFF);

    private static readonly string[] PreferredKeys =
    [
        "h", "tof", "baro", "bat",
        "pitch", "roll", "yaw",
        "vgx", "vgy", "vgz",
        "templ", "temph",
        "time", "agx", "agy", "agz",
        "mid", "x", "y", "z", "mpry",
        "sn", "sdk", "wifi", "speed", "hardware",
    ];

    private static readonly Dictionary<string, string> Labels = new()
    {
        ["h"] = "Altitude h (cm)",
        ["tof"] = "Downward distance tof (cm)",
        ["baro"] = "Barometer baro (cm)",
        ["bat"] = "Battery bat (%)",
        ["pitch"] = "Pitch (°)",
        ["roll"] = "Roll (°)",
        ["yaw"] = "Yaw (°)",
        ["vgx"] = "Velocity X vgx",
        ["vgy"] = "Velocity Y vgy",
        ["vgz"] = "Velocity Z vgz",
        ["templ"] = "Lowest temperature templ (°C)",
        ["temph"] = "Highest temperature temph (°C)",
        ["time"] = "Motor time (s)",
        ["agx"] = "Acceleration X agx",
        ["agy"] = "Acceleration Y agy",
        ["agz"] = "Acceleration Z agz",
        ["mid"] = "Mission pad ID mid",
        ["x"] = "Mission pad X (cm)",
        ["y"] = "Mission pad Y (cm)",
        ["z"] = "Mission pad Z (cm)",
        ["mpry"] = "Mission pad attitude mpry",
        ["sn"] = "Serial sn",
        ["sdk"] = "SDK version",
        ["wifi"] = "Wi-Fi SNR",
        ["speed"] = "Speed setting (cm/s)",
        ["hardware"] = "Hardware",
    };

    private readonly PythonLink _python = new();
    private readonly TelloLink _drone = new();
    private readonly CancellationTokenSource _life = new();
    private readonly DispatcherTimer _rcTimer;
    private readonly HashSet<Key> _held = new();
    private readonly HashSet<Key> _suppressed = new();
    private readonly Queue<string> _logs = new();
    private readonly int[] _frameGate = [0];

    private bool _uiBusy;
    private bool _loggedClimbBlock;
    private bool _pythonBusy;
    private bool _automatic;
    private int _maxHeightCm = 150;
    private bool _obstacle;
    private double _obstacleScore;
    private DateTime _resultAt = DateTime.MinValue;
    private PolicyAction? _policyAction;
    private Process? _bridgeProcess;
    private readonly FlightRecorder _recorder = new();
    private readonly CommandWindow _commandWindow = new();
    private byte[]? _latestJpeg;
    private int _ticks;

    public MainWindow()
    {
        InitializeComponent();
        _automatic = false;
        LoadSettingsIntoForm();
        _python.MessageReceived += message => Post(() => OnPythonMessage(message));
        _python.ConnectionChanged += up => Post(() => SetPython(up));
        _drone.Logged += text => Post(() => AppendLog(text));
        _drone.CommandSent += text => Post(() => _commandWindow.WriteLine(text));
        _drone.Changed += () => Post(RefreshFromDrone);
        _drone.JpegReady += OnJpeg;
        _rcTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(50) };
        _rcTimer.Tick += (_, _) => OnTick();
        Loaded += (_, _) =>
        {
            _rcTimer.Start();
            PublishExecutionStatus(false);
            ShowCommandWindow(activate: false);
            Activate();
        };
        _commandWindow.PreviewKeyDown += (_, e) => Window_PreviewKeyDown(this, e);
        _commandWindow.PreviewKeyUp += (_, e) => Window_PreviewKeyUp(this, e);
        UpdateButtons();
    }

    private void OnPythonMessage(JsonElement message)
    {
        if (!message.TryGetProperty("type", out var typeElement))
            return;

        switch (typeElement.GetString())
        {
            case "hello":
                AppendLog("Python vision link is up. Camera frames go out, and results come back here.");
                SetPython(true);
                break;
            case "result":
                ApplyResult(message);
                break;
            case "log":
            case "error":
                AppendLog(Text(message, "text"));
                break;
        }
    }

    private void ApplyResult(JsonElement message)
    {
        var obstacle = Bool(message, "obstacle");
        var changed = obstacle != _obstacle;
        _obstacle = obstacle;
        _obstacleScore = Double(message, "obstacle_score");
        _policyAction = ReadAction(message);
        _resultAt = DateTime.UtcNow;
        if (changed && _python.IsControlConnected)
        {
            AppendLog(obstacle
                ? "Obstacle ahead. Forward motion will hold position."
                : "Forward obstacle cleared.");
        }

        ApplyStick();
    }

    private void OnTick()
    {
        ApplyStick();
        _ticks++;
        if (_ticks % 4 == 0)
        {
            RefreshFromDrone();
            SendContext();
        }

        if (_ticks % 40 == 0)
            PublishExecutionStatus(true);
    }

    private void ApplyStick()
    {
        var snap = _drone.Snapshot;
        var speed = SpeedSlider is null ? 35 : (int)SpeedSlider.Value;
        var desired = DesiredStick(speed);
        var obstacle = ActiveObstacle;
        var decision = FlightGuard.Apply(desired, snap.Flying, snap.HeightCm, _maxHeightCm, obstacle);

        StickInText.Text = "Input  " + FormatStick(desired);
        StickOutText.Text = "Output  " + FormatStick(decision.Stick);
        StickOutText.Foreground = decision.ForwardBlocked ? BrushRed : BrushText;

        ObstacleBanner.Visibility = obstacle ? Visibility.Visible : Visibility.Collapsed;
        ObstacleBannerText.Text = decision.ForwardBlocked
            ? "Forward cancelled — holding position"
            : "Obstacle ahead — forward motion will hold position";
        if (!_python.IsControlConnected)
        {
            ObstacleStatusText.Text = "Python is not linked, so the forward obstacle check is idle. Manual sticks still fly. The downward ToF is for altitude.";
            ObstacleStatusText.Foreground = BrushMuted;
        }
        else if (obstacle)
        {
            ObstacleStatusText.Text = $"Obstacle score {_obstacleScore:0.00}. Forward and strafe are cut so the aircraft hovers. Backward and yaw still work.";
            ObstacleStatusText.Foreground = BrushRed;
        }
        else
        {
            ObstacleStatusText.Text = "Forward obstacles are judged from the camera. The downward ToF is for altitude.";
            ObstacleStatusText.Foreground = BrushMuted;
        }

        if (decision.DescendingToLimit)
        {
            HeightLimitStatusText.Text = "Above the limit — descending slowly";
            HeightLimitStatusText.Foreground = BrushAmber;
        }
        else if (decision.ClimbBlocked)
        {
            HeightLimitStatusText.Text = "At max height — climb blocked";
            HeightLimitStatusText.Foreground = BrushAmber;
        }
        else
        {
            HeightLimitStatusText.Text = "50-1000 cm. Climb stops at the limit. If the aircraft is already above it, it descends slowly.";
            HeightLimitStatusText.Foreground = BrushMuted;
        }

        if (decision.ClimbBlocked && desired.Ud > 0)
        {
            if (!_loggedClimbBlock)
            {
                _loggedClimbBlock = true;
                AppendLog($"W is blocked. Altitude is at the {_maxHeightCm} cm limit.");
            }
        }
        else
        {
            _loggedClimbBlock = false;
        }

        if (snap.Connected && snap.Flying && !_uiBusy)
        {
            // Altitude stays off the rc packet. W and S use up/down commands, which Tello actually flies.
            _drone.SendRc(decision.Stick.Lr, decision.Stick.Fb, 0, decision.Stick.Yaw);
            if (!snap.Busy && decision.Stick.Ud != 0)
                _ = _drone.NudgeAltitudeAsync(decision.Stick.Ud > 0);
        }
    }

    private Stick DesiredStick(int speed)
    {
        if (IsTyping)
            return new Stick(0, 0, 0, 0);

        if (_automatic && _policyAction is PolicyAction action)
        {
            return new Stick(
                Scale(action.Lr, speed),
                Scale(action.Fb, speed),
                Scale(action.Ud, speed),
                Scale(action.Yaw, speed));
        }

        // Mode 2: arrows translate, W/S climb, A/D yaw. Positive yaw is clockwise.
        return new Stick(
            Axis(Key.Left, Key.Right, speed),
            Axis(Key.Down, Key.Up, speed),
            Axis(Key.S, Key.W, speed),
            Axis(Key.A, Key.D, speed));
    }

    private bool ActiveObstacle =>
        _python.IsControlConnected
        && DateTime.UtcNow - _resultAt < TimeSpan.FromSeconds(1.5)
        && _obstacle;

    private void SendContext()
    {
        if (!_python.IsControlConnected)
            return;

        var snap = _drone.Snapshot;
        var telemetry = new Dictionary<string, object?>();
        foreach (var pair in snap.Telemetry)
        {
            if (double.TryParse(pair.Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var number))
                telemetry[pair.Key] = number;
            else
                telemetry[pair.Key] = pair.Value;
        }

        _ = _python.SendAsync(new
        {
            cmd = "context",
            mode = _automatic ? "auto" : "manual",
            flying = snap.Flying,
            height_cm = snap.HeightCm,
            max_height_cm = _maxHeightCm,
            telemetry,
        });
    }

    private void RefreshFromDrone()
    {
        var snap = _drone.Snapshot;
        PublishExecutionStatus(false);

        AppliedHeightText.Text = $"Applied {_maxHeightCm} cm";
        var height = snap.HeightCm;
        AltitudeValueText.Text = height is null ? "—" : $"{height} cm";
        VideoHeightText.Text = height is null
            ? $"Altitude —   /   limit {_maxHeightCm} cm"
            : $"Altitude {height} cm   /   limit {_maxHeightCm} cm";

        var telemetry = TelemetryElement(snap.Telemetry);
        ShowAltitude(telemetry, "tof", TofValueText, " cm");
        ShowAltitude(telemetry, "baro", BaroValueText, " cm");
        if (snap.Battery >= 0)
        {
            BatteryValueText.Text = $"{snap.Battery}%";
            BatteryValueText.Foreground = snap.Battery < 20 ? BrushRed : snap.Battery < 40 ? BrushAmber : BrushGreen;
        }
        else
        {
            BatteryValueText.Text = "—";
            BatteryValueText.Foreground = BrushText;
        }

        FillTelemetry(telemetry);

        if (snap.TelemetryAgeMs is int ageMs)
        {
            TelemetryAgeText.Text = ageMs > 2000 ? $"Received {ageMs} ms — stale" : $"Received {ageMs} ms";
            TelemetryAgeText.Foreground = ageMs > 2000 ? BrushAmber : BrushMuted;
        }
        else
        {
            TelemetryAgeText.Text = snap.Connected ? "Waiting for telemetry" : "Waiting for data";
            TelemetryAgeText.Foreground = BrushMuted;
        }

        if (!snap.Connected)
        {
            CameraImage.Source = null;
            VideoPlaceholder.Visibility = Visibility.Visible;
            VideoPlaceholder.Text = "Enter the drone IP and port, then press Connect drone.";
        }
        else if (CameraImage.Source is null)
        {
            VideoPlaceholder.Visibility = Visibility.Visible;
            VideoPlaceholder.Text = snap.Streaming ? "Waiting for the camera picture" : "Opening the forward camera";
        }

        if (!snap.Connected)
            FinishRecording();

        ModeStatusText.Text = _python.IsControlConnected
            ? "Manual control. Python returns vision results. This controller sends the movement commands."
            : "Manual control from this controller. Start the Python bridge when you want vision or reinforcement learning.";

        UpdateButtons();
    }

    private void FillTelemetry(JsonElement telemetry)
    {
        TelemetryList.Children.Clear();
        var seen = new HashSet<string>();
        var any = false;

        foreach (var key in PreferredKeys)
        {
            if (!telemetry.TryGetProperty(key, out var value))
                continue;
            AddTelemetryRow(key, value);
            seen.Add(key);
            any = true;
        }

        foreach (var property in telemetry.EnumerateObject().OrderBy(item => item.Name, StringComparer.Ordinal))
        {
            if (seen.Contains(property.Name))
                continue;
            AddTelemetryRow(property.Name, property.Value);
            any = true;
        }

        if (!any)
        {
            TelemetryList.Children.Add(new TextBlock
            {
                Text = "No values received yet.",
                Foreground = BrushMuted,
            });
        }
    }

    private void AddTelemetryRow(string key, JsonElement value)
    {
        var row = new Grid { Margin = new Thickness(0, 0, 0, 4) };
        row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });

        var name = new TextBlock
        {
            Text = Labels.TryGetValue(key, out var label) ? label : key,
            Foreground = key is "h" or "tof" or "baro" ? BrushAccent : BrushMuted,
            TextWrapping = TextWrapping.Wrap,
            Margin = new Thickness(0, 0, 12, 0),
        };
        var number = new TextBlock
        {
            Text = FormatValue(value),
            FontWeight = FontWeights.SemiBold,
            HorizontalAlignment = HorizontalAlignment.Right,
        };
        Grid.SetColumn(number, 1);
        row.Children.Add(name);
        row.Children.Add(number);
        TelemetryList.Children.Add(row);
    }

    private void ShowAltitude(JsonElement telemetry, string key, TextBlock target, string suffix)
    {
        if (!TryNumber(telemetry, key, out var number))
        {
            target.Text = "—";
            return;
        }

        target.Text = number % 1 == 0 ? $"{number:0}{suffix}" : $"{number:0.##}{suffix}";
    }

    private void UpdateButtons()
    {
        var snap = _drone.Snapshot;
        var python = _python.IsControlConnected;
        var bridgeProcessRunning = _bridgeProcess is { HasExited: false };
        var busy = _uiBusy || snap.Busy;
        LinkPythonButton.IsEnabled = !python && !_pythonBusy;
        UnlinkPythonButton.IsEnabled = python && !_pythonBusy;
        StartBridgeButton.IsEnabled = !bridgeProcessRunning;
        StopBridgeButton.IsEnabled = bridgeProcessRunning;
        ConnectDroneButton.IsEnabled = !snap.Connected && !busy;
        DisconnectDroneButton.IsEnabled = snap.Connected && !busy;
        TakeoffButton.IsEnabled = snap.Connected && !snap.Flying && !busy;
        LandButton.IsEnabled = snap.Flying && !busy;
        HoverButton.IsEnabled = snap.Connected && snap.Flying;
        EmergencyButton.IsEnabled = snap.Connected;
        SaveHeightButton.IsEnabled = true;
        RecordVideoButton.IsEnabled = snap.Connected || _recorder.IsRecording;
        if (!_recorder.IsRecording)
        {
            RecordVideoButton.Content = "Record video";
            RecordVideoButton.ClearValue(BackgroundProperty);
            RecordVideoButton.ClearValue(BorderBrushProperty);
        }
    }

    private void SetPython(bool up)
    {
        BridgeStatusText.Text = up
            ? $"Python linked  {_python.Endpoint}"
            : "Python idle. Manual flight does not need it.";
        BridgeStatusText.Foreground = up ? BrushGreen : BrushMuted;
        if (!up)
        {
            _obstacle = false;
            _policyAction = null;
            _resultAt = DateTime.MinValue;
        }

        UpdateButtons();
        if (IsLoaded)
        {
            ApplyStick();
            PublishExecutionStatus(false);
        }
    }

    private void OnJpeg(byte[] jpeg)
    {
        _python.SendFrame(jpeg);
        _recorder.Push(jpeg);
        _latestJpeg = jpeg;
        if (Interlocked.CompareExchange(ref _frameGate[0], 1, 0) != 0)
            return;

        Post(ShowLatestFrame);
    }

    private void ShowLatestFrame()
    {
        var jpeg = _latestJpeg;
        Interlocked.Exchange(ref _frameGate[0], 0);
        if (jpeg is null || !_drone.Snapshot.Connected)
            return;

        try
        {
            CameraImage.Source = ToBitmap(jpeg);
            VideoPlaceholder.Visibility = Visibility.Collapsed;
        }
        catch
        {
            // A partial frame is replaced by the next one.
        }
    }

    private static BitmapImage ToBitmap(byte[] jpeg)
    {
        var image = new BitmapImage();
        using var stream = new MemoryStream(jpeg);
        image.BeginInit();
        image.CacheOption = BitmapCacheOption.OnLoad;
        image.StreamSource = stream;
        image.EndInit();
        image.Freeze();
        return image;
    }

    private async void LinkPython_Click(object sender, RoutedEventArgs e)
    {
        Focus();
        if (_python.IsControlConnected || _pythonBusy)
            return;
        if (!TryReadPython(out var address, out var controlPort, out var videoPort))
            return;

        _pythonBusy = true;
        UpdateButtons();
        AppendLog($"Linking Python at {address}:{controlPort}, video port {videoPort}.");
        try
        {
            await _python.ConnectAsync(address, controlPort, videoPort, _life.Token);
            SaveSettings();
            AppendLog($"Python linked at {address}:{controlPort}, video port {videoPort}.");
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            AppendLog($"Could not link Python at {address}:{controlPort}. Start the Python server, then press Link Python. ({ex.Message})");
        }
        finally
        {
            _pythonBusy = false;
            UpdateButtons();
        }
    }

    private void UnlinkPython_Click(object sender, RoutedEventArgs e)
    {
        Focus();
        if (!_python.IsControlConnected)
            return;

        _python.Disconnect();
        AppendLog("Python unlinked. Manual flight does not need it.");
        UpdateButtons();
    }

    private void StartBridge_Click(object sender, RoutedEventArgs e)
    {
        Focus();
        StartPythonBridge();
    }

    private void StopBridge_Click(object sender, RoutedEventArgs e)
    {
        Focus();
        StopPythonBridge();
    }

    private async void ConnectDrone_Click(object sender, RoutedEventArgs e)
    {
        Focus();
        if (_uiBusy || _drone.Snapshot.Connected)
            return;
        if (!TryReadEndpoint(out var address, out var port))
            return;

        _uiBusy = true;
        UpdateButtons();
        AppendLog($"Connect started for {address}:{port}. State listen {TelloLink.StatePort}. Video listen {TelloLink.VideoPort}.");
        PublishExecutionStatus(false);
        try
        {
            await _drone.ConnectAsync(address, port, _life.Token);
            if (_drone.Snapshot.Connected)
                SaveSettings();
        }
        catch (Exception ex)
        {
            AppendLog(ex.Message);
        }
        finally
        {
            _uiBusy = false;
            RefreshFromDrone();
        }
    }

    private async void DisconnectDrone_Click(object sender, RoutedEventArgs e)
    {
        Focus();
        if (_uiBusy || !_drone.Snapshot.Connected)
            return;

        _uiBusy = true;
        UpdateButtons();
        try
        {
            await _drone.DisconnectAsync();
        }
        catch (Exception ex)
        {
            AppendLog(ex.Message);
        }
        finally
        {
            _uiBusy = false;
            RefreshFromDrone();
        }
    }

    private void Takeoff_Click(object sender, RoutedEventArgs e)
    {
        Focus();
        Takeoff();
    }

    private void Land_Click(object sender, RoutedEventArgs e)
    {
        Focus();
        Land();
    }

    private void Hover_Click(object sender, RoutedEventArgs e)
    {
        Focus();
        HoldPosition();
    }

    private void Emergency_Click(object sender, RoutedEventArgs e)
    {
        if (!_drone.Snapshot.Connected)
            return;

        var answer = MessageBox.Show(
            this,
            "The motors stop immediately. If the aircraft is airborne, it will fall.",
            "Emergency stop",
            MessageBoxButton.OKCancel,
            MessageBoxImage.Warning);
        if (answer != MessageBoxResult.OK)
            return;

        _drone.Emergency();
    }

    private void SaveHeight_Click(object sender, RoutedEventArgs e) => SaveHeight();

    private void BrowseRecordFolder_Click(object sender, RoutedEventArgs e)
    {
        Focus();
        var dialog = new Microsoft.Win32.OpenFolderDialog
        {
            Title = "Choose a folder for recorded video",
            InitialDirectory = Directory.Exists(RecordFolderBox.Text.Trim()) ? RecordFolderBox.Text.Trim() : FlightRecorder.DefaultFolder,
        };
        if (dialog.ShowDialog(this) == true)
            RecordFolderBox.Text = dialog.FolderName;
    }

    private void SaveFolder_Click(object sender, RoutedEventArgs e)
    {
        Focus();
        if (!TryReadRecordFolder(out var folder))
            return;
        if (!SaveSettings())
            return;
        AppendLog($"Saved record folder {folder}");
        RefreshRecordedVideos();
    }

    private void OpenRecordFolder_Click(object sender, RoutedEventArgs e)
    {
        Focus();
        if (!TryReadRecordFolder(out var folder))
            return;

        try
        {
            Directory.CreateDirectory(folder);
            var videos = ListRecordedVideos(folder);
            RefreshRecordedVideos(videos);
            if (videos.Count == 0)
            {
                AppendLog($"No videos in {folder}");
                Process.Start(new ProcessStartInfo
                {
                    FileName = folder,
                    UseShellExecute = true,
                });
                return;
            }

            var newest = videos[0].FullName;
            Process.Start(new ProcessStartInfo
            {
                FileName = "explorer.exe",
                Arguments = "/select,\"" + newest + "\"",
                UseShellExecute = false,
            });
        }
        catch (Exception ex)
        {
            AppendLog($"Could not open the folder. {ex.Message}");
        }
    }

    private void RecordedVideoList_MouseDoubleClick(object sender, MouseButtonEventArgs e)
    {
        if (RecordedVideoList.SelectedItem is not ListBoxItem item || item.Tag is not string path || !File.Exists(path))
            return;

        try
        {
            Process.Start(new ProcessStartInfo
            {
                FileName = path,
                UseShellExecute = true,
            });
        }
        catch (Exception ex)
        {
            AppendLog($"Could not open the video. {ex.Message}");
        }
    }

    private void RefreshRecordedVideos()
    {
        if (!TryReadRecordFolder(out var folder))
        {
            RecordedVideoList.Items.Clear();
            return;
        }

        RefreshRecordedVideos(Directory.Exists(folder) ? ListRecordedVideos(folder) : new List<FileInfo>());
    }

    private void RefreshRecordedVideos(List<FileInfo> videos)
    {
        RecordedVideoList.Items.Clear();
        if (videos.Count == 0)
        {
            RecordedVideoList.Items.Add(new ListBoxItem
            {
                Content = "No videos in this folder yet.",
                Foreground = BrushMuted,
                IsEnabled = false,
            });
            return;
        }

        foreach (var file in videos)
        {
            RecordedVideoList.Items.Add(new ListBoxItem
            {
                Content = $"{file.LastWriteTime:yyyy-MM-dd HH:mm}   {file.Name}",
                Tag = file.FullName,
            });
        }
    }

    private static List<FileInfo> ListRecordedVideos(string folder)
    {
        return new DirectoryInfo(folder)
            .EnumerateFiles()
            .Where(file => file.Extension.Equals(".avi", StringComparison.OrdinalIgnoreCase)
                || file.Extension.Equals(".mp4", StringComparison.OrdinalIgnoreCase)
                || file.Extension.Equals(".mkv", StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(file => file.LastWriteTime)
            .ToList();
    }

    private void RecordVideo_Click(object sender, RoutedEventArgs e)
    {
        Focus();
        if (_recorder.IsRecording)
        {
            FinishRecording();
            return;
        }

        if (!_drone.Snapshot.Connected)
        {
            AppendLog("Connect the drone before recording.");
            return;
        }

        if (!TryReadRecordFolder(out var folder))
            return;

        try
        {
            Directory.CreateDirectory(folder);
            var file = _recorder.Start(folder);
            RecordVideoButton.Content = "Stop recording";
            RecordVideoButton.Background = BrushRed;
            RecordVideoButton.BorderBrush = BrushRed;
            AppendLog($"Recording video to {file}");
            SaveSettings();
        }
        catch (Exception ex)
        {
            _recorder.Stop();
            AppendLog($"Could not start recording. {ex.Message}");
        }

        UpdateButtons();
    }

    private void FinishRecording()
    {
        if (!_recorder.IsRecording)
            return;

        var file = _recorder.Stop();
        if (file is null)
        {
            var error = _recorder.LastError;
            AppendLog(string.IsNullOrWhiteSpace(error)
                ? "Recording stopped. No video file was saved."
                : $"Recording stopped. {error}");
        }
        else
        {
            AppendLog($"Saved video {file}");
        }
        RefreshRecordedVideos();
        UpdateButtons();
    }

    private bool TryReadRecordFolder(out string folder)
    {
        folder = RecordFolderBox.Text.Trim();
        if (folder.Length == 0)
        {
            AppendLog("Enter a folder path for recorded video.");
            return false;
        }

        if (folder.IndexOfAny(Path.GetInvalidPathChars()) >= 0)
        {
            AppendLog("That folder path contains characters Windows does not allow.");
            return false;
        }

        return true;
    }

    private void NoteStickWhileLanded()
    {
        var snap = _drone.Snapshot;
        if (!snap.Connected)
            AppendLog("Connect the drone before using the sticks.");
        else if (!snap.Flying)
            AppendLog("Sticks move the drone only after takeoff. Press 1 to take off.");
    }

    private void Takeoff()
    {
        var snap = _drone.Snapshot;
        if (!snap.Connected || snap.Flying || snap.Busy || _uiBusy)
            return;
        _ = _drone.TakeoffAsync();
    }

    private void Land()
    {
        var snap = _drone.Snapshot;
        if (!snap.Flying || snap.Busy || _uiBusy)
            return;
        _ = _drone.LandAsync();
    }

    private void HoldPosition()
    {
        foreach (var key in _held)
            _suppressed.Add(key);
        ApplyStick();
    }

    private void SaveHeight()
    {
        if (!int.TryParse(MaxHeightBox.Text.Trim(), out var cm))
        {
            AppendLog("Enter max height as a number in centimeters.");
            return;
        }

        if (cm is < 50 or > 1000)
        {
            AppendLog("Max height must be 50-1000 cm.");
            return;
        }

        _maxHeightCm = cm;
        AppliedHeightText.Text = $"Applied {cm} cm";
        SaveSettings();
        AppendLog($"Saved max height {cm} cm.");
        Focus();
    }

    private int Axis(Key negative, Key positive, int speed)
    {
        var value = 0;
        if (Contributes(positive))
            value += speed;
        if (Contributes(negative))
            value -= speed;
        return Math.Clamp(value, -100, 100);
    }

    private bool Contributes(Key key) => _held.Contains(key) && !_suppressed.Contains(key);

    private void Window_PreviewKeyDown(object sender, KeyEventArgs e)
    {
        if (IsTyping)
            return;

        var key = NormalizeKey(e);
        if (key == Key.Space && Keyboard.FocusedElement is Button)
            return;

        if (key is Key.D1 or Key.NumPad1)
        {
            e.Handled = true;
            if (!e.IsRepeat)
                Takeoff();
            return;
        }

        if (key is Key.D2 or Key.NumPad2)
        {
            e.Handled = true;
            if (!e.IsRepeat)
                Land();
            return;
        }

        if (key == Key.Space)
        {
            e.Handled = true;
            if (!e.IsRepeat)
                HoldPosition();
            return;
        }

        if (!IsStick(key))
            return;

        if (!e.IsRepeat)
            NoteStickWhileLanded();

        _held.Add(key);
        e.Handled = true;
        ApplyStick();
    }

    private void Window_PreviewKeyUp(object sender, KeyEventArgs e)
    {
        var key = NormalizeKey(e);
        if (!_held.Remove(key) && !_suppressed.Remove(key))
            return;

        _suppressed.Remove(key);
        e.Handled = true;
        ApplyStick();
    }

    protected override void OnDeactivated(EventArgs e)
    {
        if (!FocusStillInApp())
        {
            _held.Clear();
            _suppressed.Clear();
            ApplyStick();
        }

        base.OnDeactivated(e);
    }

    private bool FocusStillInApp()
    {
        var focus = Keyboard.FocusedElement as DependencyObject;
        if (focus is null)
            return false;
        var window = Window.GetWindow(focus);
        return ReferenceEquals(window, this) || ReferenceEquals(window, _commandWindow);
    }

    private void ShowCommands_Click(object sender, RoutedEventArgs e)
    {
        Focus();
        ShowCommandWindow(activate: true);
    }

    private void ShowCommandWindow(bool activate)
    {
        if (_commandWindow.Owner is null)
            _commandWindow.Owner = this;
        if (_commandWindow.Left < 0 || _commandWindow.Top < 0 || double.IsNaN(_commandWindow.Left))
        {
            var area = SystemParameters.WorkArea;
            _commandWindow.Left = area.Left + 24;
            _commandWindow.Top = area.Top + 24;
        }

        if (!_commandWindow.IsVisible)
            _commandWindow.Show();
        if (activate)
            _commandWindow.Activate();
    }

    protected override void OnClosing(System.ComponentModel.CancelEventArgs e)
    {
        OpenSessionNotes();
        _rcTimer.Stop();
        _commandWindow.AllowClose();
        _commandWindow.Close();
        try
        {
            Task.Run(() => _drone.DisconnectAsync()).Wait(TimeSpan.FromSeconds(20));
        }
        catch
        {
            // The aircraft link is already closed.
        }

        if (_bridgeProcess is { HasExited: false })
            StopPythonBridge();
        _life.Cancel();
        _python.Dispose();
        _drone.Dispose();
        FinishRecording();
        base.OnClosing(e);
    }

    private void OpenSessionNotes()
    {
        var stamp = DateTime.Now.ToString("yyyyMMdd-HHmmss");
        OpenNote("drone-commands", stamp, _commandWindow.Transcript);
        OpenNote("execution", stamp, ExecutionLog.Transcript);
    }

    private static void OpenNote(string name, string stamp, string transcript)
    {
        try
        {
            var folder = Path.Combine(Path.GetTempPath(), "DroneSoccer");
            Directory.CreateDirectory(folder);
            var path = Path.Combine(folder, $"{name}-{stamp}.txt");
            File.WriteAllText(path, transcript.TrimEnd() + Environment.NewLine);
            Process.Start(new ProcessStartInfo
            {
                FileName = "notepad.exe",
                Arguments = $"\"{path}\"",
                UseShellExecute = true,
            });
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Could not open {name} in Notepad. {ex.Message}");
        }
    }

    private void StartPythonBridge()
    {
        if (_bridgeProcess is { HasExited: false })
            return;
        if (!TryReadPython(out var address, out var controlPort, out var videoPort))
            return;

        var root = FindProjectRoot();
        if (root is null)
        {
            AppendLog("Could not find main.py.");
            return;
        }

        var python = FindPython();
        if (python is null)
        {
            AppendLog("Could not find python.exe on PATH.");
            return;
        }

        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = python,
                    Arguments = $"-u main.py --host {address} --control-port {controlPort} --video-port {videoPort}",
                    WorkingDirectory = root,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                },
                EnableRaisingEvents = true,
            };
            process.StartInfo.Environment["PYTHONUNBUFFERED"] = "1";
            process.StartInfo.Environment["PYTHONIOENCODING"] = "utf-8";
            process.OutputDataReceived += BridgeOutput;
            process.ErrorDataReceived += BridgeOutput;
            process.Exited += (_, _) => Post(UpdateButtons);
            if (!process.Start())
            {
                AppendLog("Could not start Python.");
                return;
            }

            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            _bridgeProcess = process;
            AppendLog($"Starting the Python server at {address}:{controlPort}, video port {videoPort}. Press Link Python after it is listening.");
            UpdateButtons();
        }
        catch (Exception ex)
        {
            AppendLog($"Could not start Python. {ex.Message}");
        }
    }

    private void StopPythonBridge()
    {
        try
        {
            if (_python.IsControlConnected)
                _python.SendAsync(new { cmd = "shutdown" }).Wait(TimeSpan.FromSeconds(1));
        }
        catch
        {
            // The socket may already be closed.
        }

        var process = _bridgeProcess;
        if (process is not { HasExited: false })
            return;

        Task.Run(() =>
        {
            try
            {
                if (!process.WaitForExit(20000))
                    process.Kill(entireProcessTree: true);
            }
            catch
            {
                // The process already exited.
            }

            Post(UpdateButtons);
        });
    }

    private void BridgeOutput(object sender, DataReceivedEventArgs e)
    {
        if (string.IsNullOrWhiteSpace(e.Data))
            return;
        Post(() => AppendLog(e.Data));
    }

    private static string? FindPython()
    {
        var path = Environment.GetEnvironmentVariable("PATH");
        if (string.IsNullOrWhiteSpace(path))
            return null;

        foreach (var dir in path.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            var candidate = Path.Combine(dir.Trim(), "python.exe");
            if (File.Exists(candidate))
                return candidate;
        }

        return null;
    }

    private static string? FindProjectRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            var main = Path.Combine(dir.FullName, "main.py");
            var bridge = Path.Combine(dir.FullName, "bridge_server.py");
            if (File.Exists(main) && File.Exists(bridge))
                return dir.FullName;
            dir = dir.Parent;
        }

        return null;
    }

    private void Camera_MouseDown(object sender, MouseButtonEventArgs e) => Focus();

    private void SpeedSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (SpeedValueText is null)
            return;
        SpeedValueText.Text = ((int)e.NewValue).ToString();
    }

    private void DroneIpBox_PreviewTextInput(object sender, TextCompositionEventArgs e)
    {
        e.Handled = e.Text.Any(ch => !char.IsDigit(ch) && ch != '.');
    }

    private void DronePortBox_PreviewTextInput(object sender, TextCompositionEventArgs e)
    {
        e.Handled = e.Text.Any(ch => !char.IsDigit(ch));
    }

    private void SetIpBox(string text)
    {
        if (DroneIpBox.Text == text)
            return;
        DroneIpBox.Text = text;
    }

    private void MaxHeightBox_PreviewTextInput(object sender, TextCompositionEventArgs e)
    {
        e.Handled = e.Text.Any(ch => !char.IsDigit(ch));
    }

    private void MaxHeightBox_KeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key != Key.Enter)
            return;
        SaveHeight();
        e.Handled = true;
    }

    private void MaxHeightBox_Pasting(object sender, DataObjectPastingEventArgs e)
    {
        if (!e.DataObject.GetDataPresent(typeof(string)) || e.DataObject.GetData(typeof(string)) is not string text || text.Any(ch => !char.IsDigit(ch)))
            e.CancelCommand();
    }

    private void LoadSettingsIntoForm()
    {
        var settings = SettingsStore.Load(SettingsPath());
        _maxHeightCm = settings.MaxHeightCm;
        SetHeightBox(_maxHeightCm.ToString());
        SetIpBox(settings.DroneIp);
        DronePortBox.Text = settings.CommandPort.ToString(CultureInfo.InvariantCulture);
        PythonIpBox.Text = settings.PythonIp;
        PythonControlPortBox.Text = settings.PythonControlPort.ToString(CultureInfo.InvariantCulture);
        PythonVideoPortBox.Text = settings.PythonVideoPort.ToString(CultureInfo.InvariantCulture);
        RecordFolderBox.Text = string.IsNullOrWhiteSpace(settings.RecordFolder)
            ? FlightRecorder.DefaultFolder
            : settings.RecordFolder;
        RefreshRecordedVideos();
        AppliedHeightText.Text = $"Applied {_maxHeightCm} cm";
        LinkPortsText.Text =
            $"Command port is the aircraft UDP port (default {TelloLink.CommandPort}). This PC listens on state port {TelloLink.StatePort} and video port {TelloLink.VideoPort}.";
    }

    private bool SaveSettings()
    {
        if (!TryReadEndpoint(out var address, out var port))
            return false;

        try
        {
            var settings = SettingsStore.Load(SettingsPath());
            settings.MaxHeightCm = _maxHeightCm;
            settings.DroneIp = address.ToString();
            settings.CommandPort = port;
            settings.RecordFolder = RecordFolderBox.Text.Trim();
            if (TryReadPython(out var pythonAddress, out var controlPort, out var videoPort, report: false))
            {
                settings.PythonIp = pythonAddress.ToString();
                settings.PythonControlPort = controlPort;
                settings.PythonVideoPort = videoPort;
            }

            SettingsStore.Save(SettingsPath(), settings);
            return true;
        }
        catch (Exception ex)
        {
            AppendLog($"Could not save settings. {ex.Message}");
        }

        return false;
    }

    private string SettingsPath()
    {
        var root = FindProjectRoot() ?? AppContext.BaseDirectory;
        return Path.Combine(root, "settings.json");
    }

    private bool TryReadPython(out IPAddress address, out int controlPort, out int videoPort, bool report = true)
    {
        address = IPAddress.None;
        controlPort = 0;
        videoPort = 0;
        var ip = PythonIpBox.Text.Trim();
        if (!IPAddress.TryParse(ip, out var parsed) || parsed.AddressFamily != AddressFamily.InterNetwork)
        {
            if (report)
                AppendLog("Python IP must be an IPv4 address.");
            return false;
        }

        if (!int.TryParse(PythonControlPortBox.Text.Trim(), NumberStyles.None, CultureInfo.InvariantCulture, out controlPort) || controlPort is < 1 or > 65535)
        {
            if (report)
                AppendLog("Python result port must be 1-65535.");
            return false;
        }

        if (!int.TryParse(PythonVideoPortBox.Text.Trim(), NumberStyles.None, CultureInfo.InvariantCulture, out videoPort) || videoPort is < 1 or > 65535)
        {
            if (report)
                AppendLog("Python video port must be 1-65535.");
            return false;
        }

        if (controlPort == videoPort)
        {
            if (report)
                AppendLog("Python result port and video port must be different.");
            return false;
        }

        address = parsed;
        return true;
    }

    private bool TryReadEndpoint(out IPAddress address, out int port)
    {
        address = IPAddress.None;
        port = 0;
        var ip = DroneIpBox.Text.Trim();
        if (!IPAddress.TryParse(ip, out var parsed) || parsed.AddressFamily != AddressFamily.InterNetwork)
        {
            AppendLog("Drone IP must be an IPv4 address.");
            return false;
        }

        if (!int.TryParse(DronePortBox.Text.Trim(), NumberStyles.None, CultureInfo.InvariantCulture, out port) || port is < 1 or > 65535)
        {
            AppendLog("Drone port must be 1-65535.");
            return false;
        }

        address = parsed;
        return true;
    }

    private void SetHeightBox(string text)
    {
        if (MaxHeightBox.Text == text)
            return;
        MaxHeightBox.Text = text;
    }

    private void AppendLog(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return;

        _logs.Enqueue($"{DateTime.Now:HH:mm:ss}  {text}");
        while (_logs.Count > 8)
            _logs.Dequeue();
        LogText.Text = string.Join(Environment.NewLine, _logs);
        ExecutionLog.WriteLine(text);
    }

    private void PublishExecutionStatus(bool repeat)
    {
        var snap = _drone.Snapshot;
        string phase;
        if (snap.Connected)
            phase = "CONNECTED";
        else if (snap.Busy || _uiBusy)
            phase = "CONNECTING";
        else if (LooksLikeFailure(snap.Message))
            phase = "FAILED";
        else
            phase = "NOT CONNECTED";

        DroneStatusText.Text = phase switch
        {
            "CONNECTED" => $"CONNECTED  {snap.Endpoint}",
            "CONNECTING" when snap.Endpoint.Length > 0 => $"CONNECTING  {snap.Endpoint}",
            "CONNECTING" => "CONNECTING",
            _ => "NOT CONNECTED",
        };
        DroneStatusText.Foreground = phase switch
        {
            "CONNECTED" => BrushGreen,
            "CONNECTING" => BrushAmber,
            "FAILED" => BrushRed,
            _ => BrushMuted,
        };

        var python = _python.IsControlConnected ? "python vision linked" : "python idle";
        var video = snap.FrameReady ? "video yes" : snap.Streaming ? "video opening" : "video no";
        var flying = snap.Flying ? "flying" : "on the ground";
        var battery = snap.Battery >= 0 ? $"battery {snap.Battery}%" : "battery —";
        var altitude = snap.HeightCm is int cm ? $"altitude {cm} cm" : "altitude —";
        var state = snap.TelemetryAgeMs is int age ? $"state {age} ms" : "state none";
        var endpoint = snap.Endpoint.Length > 0 ? snap.Endpoint : "no aircraft endpoint";
        var detail = phase == "FAILED" && !string.IsNullOrWhiteSpace(snap.Message)
            ? snap.Message
            : $"{endpoint}   {battery}   {altitude}   {video}   {flying}   {state}   {python}";
        ExecutionLog.SetStatus(phase, detail, repeat);
    }

    private static bool LooksLikeFailure(string message)
    {
        if (string.IsNullOrWhiteSpace(message) || message is "Drone idle" or "Drone disconnected.")
            return false;
        return message.Contains("fail", StringComparison.OrdinalIgnoreCase)
            || message.Contains("cancelled", StringComparison.OrdinalIgnoreCase)
            || message.Contains("already in use", StringComparison.OrdinalIgnoreCase)
            || message.Contains("did not", StringComparison.OrdinalIgnoreCase);
    }

    private void Post(Action action)
    {
        if (_life.IsCancellationRequested)
            return;
        Dispatcher.BeginInvoke(action);
    }

    private bool IsTyping => Keyboard.FocusedElement is TextBox box && !box.IsReadOnly;

    private static bool IsStick(Key key) => key is Key.Up or Key.Down or Key.Left or Key.Right
        or Key.W or Key.A or Key.S or Key.D;

    private static Key NormalizeKey(KeyEventArgs e)
    {
        if (e.Key == Key.ImeProcessed)
            return e.ImeProcessedKey;
        if (e.Key == Key.System)
            return e.SystemKey;
        return e.Key;
    }

    private static string FormatStick(Stick stick) =>
        $"L/R {stick.Lr}   F/B {stick.Fb}   U/D {stick.Ud}   Yaw {stick.Yaw}";

    private static int Scale(double unit, int speed) =>
        FlightGuard.ClampAxis((int)Math.Round(Math.Clamp(unit, -1, 1) * speed));

    private static PolicyAction? ReadAction(JsonElement message)
    {
        if (!message.TryGetProperty("action", out var action) || action.ValueKind != JsonValueKind.Object)
            return null;
        return new PolicyAction(Unit(action, "lr"), Unit(action, "fb"), Unit(action, "ud"), Unit(action, "yaw"));
    }

    private static double Unit(JsonElement action, string name)
    {
        if (!action.TryGetProperty(name, out var element) || element.ValueKind != JsonValueKind.Number)
            return 0;
        return element.GetDouble();
    }

    private static JsonElement TelemetryElement(IReadOnlyDictionary<string, string> telemetry)
    {
        var obj = new JsonObject();
        foreach (var pair in telemetry)
        {
            if (double.TryParse(pair.Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var number))
                obj[pair.Key] = number;
            else
                obj[pair.Key] = pair.Value;
        }

        return JsonSerializer.SerializeToElement(obj);
    }

    private static bool Bool(JsonElement message, string name)
    {
        return message.TryGetProperty(name, out var element) && element.ValueKind == JsonValueKind.True;
    }

    private static double Double(JsonElement message, string name)
    {
        if (!message.TryGetProperty(name, out var element) || element.ValueKind != JsonValueKind.Number)
            return 0;
        return element.GetDouble();
    }

    private static string Text(JsonElement message, string name)
    {
        if (!message.TryGetProperty(name, out var element) || element.ValueKind != JsonValueKind.String)
            return "";
        return element.GetString() ?? "";
    }

    private static bool TryNumber(JsonElement parent, string name, out double number)
    {
        number = 0;
        if (!parent.TryGetProperty(name, out var element) || element.ValueKind != JsonValueKind.Number)
            return false;
        number = element.GetDouble();
        return true;
    }

    private static string FormatValue(JsonElement element)
    {
        return element.ValueKind switch
        {
            JsonValueKind.Number when element.TryGetInt64(out var integer) => integer.ToString(),
            JsonValueKind.Number => element.GetDouble().ToString("0.##"),
            JsonValueKind.String => element.GetString() ?? "",
            JsonValueKind.True => "true",
            JsonValueKind.False => "false",
            JsonValueKind.Null => "-",
            _ => element.ToString(),
        };
    }

    private static SolidColorBrush MakeBrush(byte red, byte green, byte blue)
    {
        var brush = new SolidColorBrush(Color.FromRgb(red, green, blue));
        brush.Freeze();
        return brush;
    }

    private readonly record struct PolicyAction(double Lr, double Fb, double Ud, double Yaw);
}
