using System.Globalization;
using System.Net;
using System.Net.NetworkInformation;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using OpenCvSharp;

namespace DroneSoccerWPF;

public sealed record TelloSnapshot(
    bool Connected,
    bool Flying,
    bool Busy,
    bool Streaming,
    bool FrameReady,
    int Battery,
    int? HeightCm,
    int? TelemetryAgeMs,
    string Message,
    string Endpoint,
    IReadOnlyDictionary<string, string> Telemetry);

/// <summary>
/// Direct Tello EDU session. Command UDP goes to the aircraft port (8889 by default).
/// State arrives on local port 8890. Video arrives on local port 11111.
/// </summary>
public sealed class TelloLink : IDisposable
{
    public const int CommandPort = 8889;
    public const int LocalCommandPort = 9000;
    public const int StatePort = 8890;
    public const int VideoPort = 11111;
    private const int DecoderPort = 11112;

    private readonly object _gate = new();
    private readonly object _sendLock = new();
    private readonly SemaphoreSlim _commands = new(1, 1);
    private readonly Dictionary<string, string> _telemetry = new(StringComparer.Ordinal);

    private UdpClient? _command;
    private UdpClient? _state;
    private CancellationTokenSource? _sessionCts;
    private TaskCompletionSource<string>? _pending;
    private VideoCapture? _capture;
    private UdpClient? _videoIn;
    private Task? _videoTask;
    private int _videoPacketCount;
    private int _awaiting;
    private int _loggedState;
    private int _loggedVideo;
    private string _lastTracedRc = "";
    private bool _connected;
    private bool _flying;
    private bool _busy;
    private int _altitudeNudge;
    private bool _streaming;
    private bool _frameReady;
    private bool _emergency;
    private string _message = "Drone idle";
    private string _endpoint = "";
    private DateTime _stateAtUtc;
    private IPEndPoint? _drone;

    public event Action<string>? Logged;
    public event Action<string>? CommandSent;
    public event Action? Changed;
    public event Action<byte[]>? JpegReady;

    public TelloSnapshot Snapshot
    {
        get
        {
            lock (_gate)
            {
                int? age = _stateAtUtc == DateTime.MinValue
                    ? null
                    : (int)(DateTime.UtcNow - _stateAtUtc).TotalMilliseconds;
                int? height = ReadCm(_telemetry, "h");
                var battery = ReadCm(_telemetry, "bat") ?? -1;
                return new TelloSnapshot(
                    _connected,
                    _flying,
                    _busy,
                    _streaming,
                    _frameReady,
                    battery,
                    height,
                    age,
                    _message,
                    _endpoint,
                    new Dictionary<string, string>(_telemetry));
            }
        }
    }

    public async Task ConnectAsync(IPAddress address, int port, CancellationToken cancellationToken)
    {
        await DisconnectAsync().ConfigureAwait(false);

        var session = new CancellationTokenSource();
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(session.Token, cancellationToken);
        var token = linked.Token;
        UdpClient? command = null;
        UdpClient? state = null;
        string? error = null;
        try
        {
            command = OpenCommandSocket();
            command.Connect(address, port);
            state = new UdpClient(StatePort);
            Log($"Command UDP {command.Client.LocalEndPoint} -> {address}:{port}");
            Log($"State UDP listening on 0.0.0.0:{StatePort}");

            lock (_gate)
            {
                _command = command;
                _state = state;
                _sessionCts = session;
                _drone = new IPEndPoint(address, port);
                _endpoint = $"{address}:{port}";
                _emergency = false;
                _busy = true;
                _message = $"Connecting to the drone at {_endpoint}.";
            }

            Publish();
            Log(_message);

            _ = Task.Run(() => CommandReceiveLoop(command, session.Token), session.Token);
            _ = Task.Run(() => StateReceiveLoop(state, session.Token), session.Token);

            var ready = await CommandRetryAsync("command", 3, TimeSpan.FromSeconds(7), token).ConfigureAwait(false);
            if (!ready.Equals("ok", StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException($"The drone answered \"{ready}\".");

            lock (_gate)
            {
                _connected = true;
                _message = "Drone connected. Opening the video stream.";
            }

            Publish();
            Log("Drone connected. Opening the video stream.");
            // Bind UDP 11111 before streamon. The aircraft sends the H.264 headers once, and a late listener never gets a picture.
            var localVideo = LocalAddressToward(address, port, (IPEndPoint)command.Client.LocalEndPoint!);
            if (!StartVideo(session, localVideo))
            {
                lock (_gate)
                    _message = "Drone connected, but the camera port could not be opened. Manual sticks still work.";
                Publish();
            }
            else
            {
            await ReadIdentityAsync(token).ConfigureAwait(false);
            var listening = await WaitForVideoPortAsync(token).ConfigureAwait(false);
            if (!listening)
                Log($"Video decoder has not bound UDP {VideoPort} yet. The first camera packets may be missed.");
            try
            {
                var stream = await CommandRetryAsync("streamon", 2, TimeSpan.FromSeconds(5), token).ConfigureAwait(false);
                if (!stream.Equals("ok", StringComparison.OrdinalIgnoreCase))
                    throw new InvalidOperationException(stream);
                lock (_gate)
                {
                    _streaming = true;
                    _message = "Drone connected. Waiting for the camera picture.";
                }

                Log("Video stream requested. Waiting for the first camera frame.");
                _ = RecoverVideoAsync(session.Token);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                lock (_gate)
                    _message = "Connected, but there is no forward video. Manual sticks still work.";
                Log($"Connected, but the video stream did not start. Manual sticks still work. ({ex.Message})");
            }
            }

            _ = Task.Run(() => KeepAliveLoop(session.Token), session.Token);
        }
        catch (Exception ex)
        {
            error = ex is SocketException { SocketErrorCode: SocketError.AddressAlreadyInUse }
                ? $"State port {StatePort} is already in use. Close the other program that is listening for the aircraft."
                : ex is OperationCanceledException
                    ? "Drone connection cancelled."
                    : $"Drone connection failed at {address}:{port}. Confirm the PC can reach that address. ({ex.Message})";
            try { command?.Close(); } catch { /* already closed */ }
            try { state?.Close(); } catch { /* already closed */ }
            CancellationTokenSource? assigned;
            lock (_gate)
                assigned = _sessionCts;
            if (!ReferenceEquals(assigned, session))
            {
                session.Cancel();
                session.Dispose();
            }

            AbortSession();
        }
        finally
        {
            lock (_gate)
            {
                _busy = false;
                if (error is not null)
                    _message = error;
            }

            if (error is not null)
            {
                Log(error);
                Log("NOT CONNECTED.");
            }
            else
                Log("CONNECTED.");
            Publish();
        }
    }

    public async Task TakeoffAsync()
    {
        if (!ReadyForCommand("Press Connect drone first."))
            return;

        lock (_gate)
        {
            if (_flying || _busy)
                return;
            if (ReadCm(_telemetry, "bat") is int battery && battery < 10)
            {
                _message = "Battery is below 10%. Climbing off the ground is blocked.";
                Publish();
                Log(_message);
                return;
            }

            _busy = true;
        }

        Publish();
        try
        {
            var token = SessionToken();
            var response = await CommandRetryAsync("takeoff", 1, TimeSpan.FromSeconds(15), token).ConfigureAwait(false);
            if (!response.Equals("ok", StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException(response);
            lock (_gate)
            {
                _flying = true;
                _message = "Airborne. Arrows move, W and S climb, A and D turn. Press 2 to land.";
            }

            Log("Airborne. Arrows move, W and S climb, A and D turn. Press 2 to land.");
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            lock (_gate)
                _message = "Could not leave the ground.";
            Log($"Could not leave the ground: {ex.Message}");
        }
        finally
        {
            lock (_gate)
                _busy = false;
            Publish();
        }
    }

    public async Task LandAsync()
    {
        bool flying;
        lock (_gate)
            flying = _connected && _flying && !_busy;
        if (!flying)
            return;

        lock (_gate)
            _busy = true;
        Publish();
        try
        {
            await LandCoreAsync(SessionToken()).ConfigureAwait(false);
        }
        finally
        {
            lock (_gate)
                _busy = false;
            Publish();
        }
    }

    public void Emergency()
    {
        UdpClient? command;
        lock (_gate)
        {
            command = _command;
            if (!_connected || command is null)
            {
                Log("No drone is connected.");
                return;
            }

            _emergency = true;
            _flying = false;
            _message = "Emergency stop cut the motors. If the aircraft was airborne, it will fall.";
            _pending?.TrySetCanceled();
        }

        try
        {
            SendBytes(command, "emergency");
            Log("Emergency stop cut the motors. If the aircraft was airborne, it will fall.");
        }
        catch (Exception ex)
        {
            Log($"Emergency stop failed: {ex.Message}");
        }

        Publish();
    }

    public void SendRc(int lr, int fb, int ud, int yaw)
    {
        var nudging = Volatile.Read(ref _altitudeNudge) != 0;
        if (Volatile.Read(ref _awaiting) != 0 && !nudging)
            return;

        UdpClient? command;
        lock (_gate)
        {
            if (!_connected || !_flying || _emergency)
                return;
            if (_busy && _altitudeNudge == 0)
                return;
            command = _command;
        }

        if (command is null)
            return;

        try
        {
            SendBytes(command, $"rc {FlightGuard.ClampAxis(lr)} {FlightGuard.ClampAxis(fb)} {FlightGuard.ClampAxis(ud)} {FlightGuard.ClampAxis(yaw)}");
        }
        catch (Exception ex)
        {
            Log($"Control command failed: {ex.Message}");
        }
    }

    /// <summary>
    /// Tello often ignores the up/down channel inside rc. A short up/down move is a real altitude change.
    /// </summary>
    public Task NudgeAltitudeAsync(bool climb)
    {
        lock (_gate)
        {
            if (!_connected || !_flying || _busy || _emergency)
                return Task.CompletedTask;
            _busy = true;
            _altitudeNudge = 1;
        }

        Publish();
        return NudgeAltitudeCoreAsync(climb);
    }

    private async Task NudgeAltitudeCoreAsync(bool climb)
    {
        var command = climb ? "up 30" : "down 30";
        try
        {
            var response = await CommandRetryAsync(command, 1, TimeSpan.FromSeconds(8), SessionToken()).ConfigureAwait(false);
            if (!response.Equals("ok", StringComparison.OrdinalIgnoreCase))
                Log($"Altitude move was refused: {response}");
        }
        catch (OperationCanceledException)
        {
            // Shutting down.
        }
        catch (Exception ex)
        {
            Log($"Altitude move failed: {ex.Message}");
        }
        finally
        {
            lock (_gate)
            {
                _busy = false;
                _altitudeNudge = 0;
            }

            Publish();
        }
    }

    public async Task DisconnectAsync()
    {
        CancellationTokenSource? session;
        bool land;
        bool wasConnected;
        lock (_gate)
        {
            session = _sessionCts;
            wasConnected = _connected;
            land = _connected && _flying;
            _busy = _connected;
        }

        if (!wasConnected && session is null)
        {
            lock (_gate)
                _busy = false;
            return;
        }

        try
        {
            if (land)
            {
                try
                {
                    await LandCoreAsync(session?.Token ?? CancellationToken.None).ConfigureAwait(false);
                }
                catch (Exception ex) when (ex is not OperationCanceledException)
                {
                    Log($"Landing during shutdown failed: {ex.Message}");
                }
            }

            bool streaming;
            UdpClient? command;
            lock (_gate)
            {
                streaming = _streaming;
                command = _command;
            }

            if (streaming && command is not null)
            {
                try
                {
                    SendBytes(command, "streamoff");
                }
                catch
                {
                    // The socket may already be closed.
                }
            }
        }
        finally
        {
            AbortSession();
            lock (_gate)
            {
                _busy = false;
                _message = wasConnected ? "Drone disconnected." : "Drone idle";
            }

            Publish();
            if (wasConnected)
            {
                Log("Drone disconnected.");
                Log("NOT CONNECTED.");
            }
        }
    }

    public void Dispose()
    {
        AbortSession();
    }

    private async Task LandCoreAsync(CancellationToken cancellationToken)
    {
        var response = await CommandRetryAsync("land", 1, TimeSpan.FromSeconds(15), cancellationToken).ConfigureAwait(false);
        if (!response.Equals("ok", StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException(response);
        lock (_gate)
        {
            _flying = false;
            _message = "Landed.";
        }

        Log("Landed.");
        Publish();
    }

    private async Task ReadIdentityAsync(CancellationToken cancellationToken)
    {
        foreach (var pair in new (string Key, string Command)[]
        {
            ("sn", "sn?"),
            ("sdk", "sdk?"),
            ("wifi", "wifi?"),
            ("speed", "speed?"),
        })
        {
            try
            {
                var value = await CommandRetryAsync(pair.Command, 1, TimeSpan.FromSeconds(2), cancellationToken).ConfigureAwait(false);
                lock (_gate)
                    _telemetry[pair.Key] = value.Trim();
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception ex)
            {
                lock (_gate)
                    _telemetry[pair.Key] = $"query failed ({ex.GetType().Name})";
            }
        }

        Publish();
    }

    private async Task KeepAliveLoop(CancellationToken cancellationToken)
    {
        using var timer = new PeriodicTimer(TimeSpan.FromSeconds(8));
        try
        {
            while (await timer.WaitForNextTickAsync(cancellationToken).ConfigureAwait(false))
            {
                bool idle;
                lock (_gate)
                    idle = _connected && !_flying && !_busy && !_emergency;
                if (!idle)
                    continue;
                try
                {
                    await CommandRetryAsync("command", 1, TimeSpan.FromSeconds(5), cancellationToken).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Log($"Keepalive failed: {ex.Message}");
                }
            }
        }
        catch (OperationCanceledException)
        {
            // Shutting down.
        }
    }

    private UdpClient OpenCommandSocket()
    {
        try
        {
            return new UdpClient(LocalCommandPort);
        }
        catch (SocketException)
        {
            var note = $"Local command port {LocalCommandPort} is busy. Using a temporary port. Wait 20 seconds before the next connect if the drone stops answering.";
            Log(note);
            NoteCommand(note);
            return new UdpClient(0);
        }
    }

    private static IPAddress LocalAddressToward(IPAddress remote, int remotePort, IPEndPoint bound)
    {
        if (!bound.Address.Equals(IPAddress.Any) && !bound.Address.Equals(IPAddress.IPv6Any))
            return bound.Address;

        using var probe = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
        probe.Connect(remote, remotePort);
        return ((IPEndPoint)probe.LocalEndPoint!).Address;
    }

    private bool StartVideo(CancellationTokenSource session, IPAddress localAddress)
    {
        UdpClient? socket = null;
        try
        {
            socket = new UdpClient();
            socket.Client.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
            try
            {
                socket.ExclusiveAddressUse = false;
            }
            catch (Exception)
            {
                // Some stacks reject this before bind. Bind still proceeds.
            }

            socket.Client.ReceiveBufferSize = 1024 * 1024;
            socket.Client.Bind(new IPEndPoint(localAddress, VideoPort));
        }
        catch (SocketException ex)
        {
            try { socket?.Close(); } catch { /* not bound */ }
            var note = $"Could not listen on {localAddress}:{VideoPort} ({ex.SocketErrorCode}). A previous controller still holds the camera port, so this connect stays up without video.";
            Log(note);
            NoteCommand(note);
            return false;
        }

        lock (_gate)
        {
            _videoIn = socket;
            _videoPacketCount = 0;
        }

        var listening = $"Video UDP listening on {localAddress}:{VideoPort}.";
        Log(listening);
        NoteCommand(listening);
        _videoTask = Task.Run(() => VideoLoop(socket!, session.Token), session.Token);
        return true;
    }

    private async Task<bool> WaitForVideoPortAsync(CancellationToken cancellationToken)
    {
        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(3);
        while (DateTime.UtcNow < deadline)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (UdpPortIsOpen(VideoPort))
                return true;
            await Task.Delay(50, cancellationToken).ConfigureAwait(false);
        }

        return UdpPortIsOpen(VideoPort);
    }

    private static bool UdpPortIsOpen(int port)
    {
        try
        {
            return IPGlobalProperties.GetIPGlobalProperties()
                .GetActiveUdpListeners()
                .Any(endpoint => endpoint.Port == port);
        }
        catch (NetworkInformationException)
        {
            return false;
        }
    }

    private async Task RecoverVideoAsync(CancellationToken cancellationToken)
    {
        try
        {
            await Task.Delay(TimeSpan.FromSeconds(4), cancellationToken).ConfigureAwait(false);
            if (Snapshot.FrameReady)
            {
                Log("Drone connected. Arrows move, W and S climb, A and D turn. Press 1 to take off.");
                return;
            }

            var packets = Volatile.Read(ref _videoPacketCount);
            var waiting = packets == 0
                ? $"No camera packets on UDP {VideoPort} yet. Requesting the video stream again."
                : $"Camera packets are arriving ({packets}), but no picture yet. Requesting the video stream again.";
            Log(waiting);
            NoteCommand(waiting);
            await CommandRetryAsync("streamon", 1, TimeSpan.FromSeconds(5), cancellationToken).ConfigureAwait(false);
            await Task.Delay(TimeSpan.FromSeconds(4), cancellationToken).ConfigureAwait(false);
            if (Snapshot.FrameReady)
            {
                Log("Drone connected. Arrows move, W and S climb, A and D turn. Press 1 to take off.");
                return;
            }

            packets = Volatile.Read(ref _videoPacketCount);
            var still = packets == 0
                ? $"Still no camera packets on UDP {VideoPort}. streamon was accepted, but the picture is not arriving."
                : $"Still no picture after {packets} camera packets on UDP {VideoPort}.";
            Log(still);
            NoteCommand(still);
            lock (_gate)
                _message = "Connected, but the camera picture has not arrived. Manual sticks still work.";
            Publish();
        }
        catch (OperationCanceledException)
        {
            // Disconnect closed the session.
        }
        catch (Exception ex)
        {
            Log($"Could not restart the video stream. ({ex.Message})");
        }
    }

    private void VideoLoop(UdpClient incoming, CancellationToken cancellationToken)
    {
        using var forward = new UdpClient();
        var relay = Task.Run(() => RelayVideo(incoming, forward, cancellationToken), cancellationToken);
        // The decoder listens on localhost. The relay owns the aircraft port, so streamon cannot outrun the socket.
        var url = $"udp://@127.0.0.1:{DecoderPort}?overrun_nonfatal=1&fifo_size=65536";
        var announcedFailure = false;
        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                VideoCapture? capture = null;
                try
                {
                    capture = new VideoCapture();
                    var opened = capture.Open(url, VideoCaptureAPIs.FFMPEG);
                    if (!opened)
                        opened = capture.Open(url);
                    if (!opened)
                    {
                        if (!announcedFailure)
                        {
                            var failed = $"Could not open the video decoder on UDP port {DecoderPort}. Retrying.";
                            Log(failed);
                            NoteCommand(failed);
                            announcedFailure = true;
                        }

                        capture.Dispose();
                        capture = null;
                        if (cancellationToken.WaitHandle.WaitOne(400))
                            return;
                        continue;
                    }

                    announcedFailure = false;
                    var ready = $"Video decoder listening on 127.0.0.1:{DecoderPort}.";
                    Log(ready);
                    NoteCommand(ready);
                    lock (_gate)
                        _capture = capture;

                    using var frame = new Mat();
                    while (!cancellationToken.IsCancellationRequested)
                    {
                        if (!capture.Read(frame) || frame.Empty())
                        {
                            if (cancellationToken.WaitHandle.WaitOne(5))
                                return;
                            continue;
                        }

                        using var fitted = Fit(frame);
                        Cv2.ImEncode(".jpg", fitted, out var jpeg, new ImageEncodingParam(ImwriteFlags.JpegQuality, 60));
                        if (jpeg is null || jpeg.Length == 0)
                            continue;

                        lock (_gate)
                            _frameReady = true;
                        if (Interlocked.Exchange(ref _loggedVideo, 1) == 0)
                        {
                            var seen = $"Video frame decoded from UDP {VideoPort}.";
                            Log(seen);
                            NoteCommand(seen);
                            lock (_gate)
                                _message = "Drone connected. Arrows move, W and S climb, A and D turn. Press 1 to take off.";
                            Publish();
                        }

                        JpegReady?.Invoke(jpeg);
                    }
                }
                catch (Exception ex) when (ex is not OperationCanceledException)
                {
                    var stopped = $"Video stopped: {ex.Message}";
                    Log(stopped);
                    NoteCommand(stopped);
                    if (cancellationToken.WaitHandle.WaitOne(400))
                        return;
                }
                finally
                {
                    lock (_gate)
                    {
                        if (ReferenceEquals(_capture, capture))
                            _capture = null;
                        _frameReady = false;
                    }

                    capture?.Release();
                    capture?.Dispose();
                }
            }
        }
        finally
        {
            try
            {
                incoming.Close();
            }
            catch (SocketException)
            {
                // The relay is already closing.
            }

            try
            {
                relay.Wait(TimeSpan.FromSeconds(1));
            }
            catch
            {
                // The relay already stopped.
            }
        }
    }

    private void RelayVideo(UdpClient incoming, UdpClient forward, CancellationToken cancellationToken)
    {
        var remote = new IPEndPoint(IPAddress.Any, 0);
        var dest = new IPEndPoint(IPAddress.Loopback, DecoderPort);
        var pending = new Queue<byte[]>();
        var decoderUp = false;
        var announced = 0;
        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                byte[] packet;
                try
                {
                    packet = incoming.Receive(ref remote);
                }
                catch (ObjectDisposedException)
                {
                    break;
                }
                catch (SocketException)
                {
                    break;
                }

                Interlocked.Increment(ref _videoPacketCount);
                if (Interlocked.Exchange(ref announced, 1) == 0)
                {
                    var kind = packet.Length > 0 && packet[0] == 0x47 ? "MPEG-TS" : "packet";
                    var line = $"Camera packet received on UDP {VideoPort} ({packet.Length} bytes, {kind}).";
                    Log(line);
                    NoteCommand(line);
                }

                if (!decoderUp)
                    decoderUp = UdpPortIsOpen(DecoderPort);
                if (!decoderUp)
                {
                    if (pending.Count < 2000)
                        pending.Enqueue(packet);
                    continue;
                }

                while (pending.Count > 0)
                {
                    var buffered = pending.Dequeue();
                    forward.Send(buffered, buffered.Length, dest);
                }

                forward.Send(packet, packet.Length, dest);
            }
        }
        catch (ObjectDisposedException)
        {
            // Shutting down.
        }
        catch (SocketException)
        {
            // Shutting down.
        }
    }

    private static Mat Fit(Mat frame)
    {
        if (frame.Width <= 960)
            return frame.Clone();

        var height = Math.Max(1, (int)(frame.Height * (960.0 / frame.Width)));
        var fitted = new Mat();
        Cv2.Resize(frame, fitted, new Size(960, height), interpolation: InterpolationFlags.Area);
        return fitted;
    }

    private void StopVideo()
    {
        VideoCapture? capture;
        UdpClient? incoming;
        lock (_gate)
        {
            capture = _capture;
            incoming = _videoIn;
            _videoIn = null;
        }

        try
        {
            incoming?.Close();
        }
        catch (SocketException)
        {
            // The relay is already closing.
        }

        try
        {
            capture?.Release();
        }
        catch
        {
            // The reader is already closing.
        }

        try
        {
            _videoTask?.Wait(TimeSpan.FromSeconds(2));
        }
        catch
        {
            // The reader already stopped.
        }
    }

    private async Task<string> CommandRetryAsync(string command, int attempts, TimeSpan timeout, CancellationToken cancellationToken)
    {
        Exception? last = null;
        for (var attempt = 0; attempt < attempts; attempt++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                var response = await CommandAsync(command, timeout, cancellationToken).ConfigureAwait(false);
                if (response.StartsWith("error", StringComparison.OrdinalIgnoreCase) || response.Contains("unknown command", StringComparison.OrdinalIgnoreCase))
                    throw new InvalidOperationException(response);
                return response;
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                last = ex;
                var note = $"No answer to \"{command}\" (attempt {attempt + 1} of {attempts}). {ex.Message}";
                if (TraceCommand(command))
                    Log(note);
                NoteCommand(note);
            }
        }

        throw last ?? new InvalidOperationException($"The drone did not answer \"{command}\".");
    }

    private async Task<string> CommandAsync(string command, TimeSpan timeout, CancellationToken cancellationToken)
    {
        await _commands.WaitAsync(cancellationToken).ConfigureAwait(false);
        Interlocked.Increment(ref _awaiting);
        try
        {
            UdpClient? commandSocket;
            lock (_gate)
                commandSocket = _command;
            if (commandSocket is null)
                throw new InvalidOperationException("The drone socket is closed.");

            var pending = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
            lock (_gate)
            {
                _pending?.TrySetCanceled();
                _pending = pending;
            }

            var trace = TraceCommand(command);
            try
            {
                if (trace)
                    Log($"TX {command}");
                SendBytes(commandSocket, command);
                var response = await pending.Task.WaitAsync(timeout, cancellationToken).ConfigureAwait(false);
                if (trace)
                    Log($"RX {response}");
                NoteCommand($"RX {response}");
                return response;
            }
            catch (TimeoutException)
            {
                throw new TimeoutException($"The drone did not answer \"{command}\".");
            }
            finally
            {
                lock (_gate)
                {
                    if (ReferenceEquals(_pending, pending))
                        _pending = null;
                }
            }
        }
        finally
        {
            Interlocked.Decrement(ref _awaiting);
            _commands.Release();
        }
    }

    private async Task CommandReceiveLoop(UdpClient client, CancellationToken cancellationToken)
    {
        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                var result = await client.ReceiveAsync(cancellationToken).ConfigureAwait(false);
                var text = Encoding.UTF8.GetString(result.Buffer).Trim('\0', ' ', '\r', '\n');
                if (text.Length == 0)
                    continue;
                TaskCompletionSource<string>? pending;
                lock (_gate)
                    pending = _pending;
                pending?.TrySetResult(text);
            }
        }
        catch (OperationCanceledException)
        {
            // Shutting down.
        }
        catch (ObjectDisposedException)
        {
            // Shutting down.
        }
        catch (SocketException)
        {
            // Shutting down.
        }
    }

    private async Task StateReceiveLoop(UdpClient client, CancellationToken cancellationToken)
    {
        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                var result = await client.ReceiveAsync(cancellationToken).ConfigureAwait(false);
                var text = Encoding.UTF8.GetString(result.Buffer).Trim('\0', ' ', '\r', '\n');
                if (!text.Contains(':', StringComparison.Ordinal))
                    continue;

                lock (_gate)
                {
                    foreach (var part in text.Split(';', StringSplitOptions.RemoveEmptyEntries))
                    {
                        var split = part.Split(':', 2);
                        if (split.Length != 2)
                            continue;
                        _telemetry[split[0].Trim()] = split[1].Trim();
                    }

                    _stateAtUtc = DateTime.UtcNow;
                }

                if (Interlocked.Exchange(ref _loggedState, 1) == 0)
                    Log($"State packet received on UDP {StatePort}.");
            }
        }
        catch (OperationCanceledException)
        {
            // Shutting down.
        }
        catch (ObjectDisposedException)
        {
            // Shutting down.
        }
        catch (SocketException)
        {
            // Shutting down.
        }
    }

    private bool TraceCommand(string command)
    {
        if (command.StartsWith("rc ", StringComparison.Ordinal))
            return false;
        lock (_gate)
            return command != "command" || !_connected;
    }

    private void SendBytes(UdpClient client, string command)
    {
        var bytes = Encoding.UTF8.GetBytes(command);
        lock (_sendLock)
            client.Send(bytes, bytes.Length);
        NoteOutbound(command);
    }

    private void NoteOutbound(string command)
    {
        if (command.StartsWith("rc ", StringComparison.Ordinal))
        {
            lock (_gate)
            {
                if (command == _lastTracedRc)
                    return;
                _lastTracedRc = command;
            }
        }

        NoteCommand($"TX {command}");
    }

    private void NoteCommand(string text) => CommandSent?.Invoke(text);

    private bool ReadyForCommand(string error)
    {
        lock (_gate)
        {
            if (_connected && _command is not null)
                return true;
        }

        Log(error);
        return false;
    }

    private CancellationToken SessionToken()
    {
        lock (_gate)
            return _sessionCts?.Token ?? CancellationToken.None;
    }

    private void AbortSession()
    {
        CancellationTokenSource? session;
        lock (_gate)
        {
            session = _sessionCts;
            _sessionCts = null;
            _connected = false;
            _flying = false;
            _streaming = false;
            _frameReady = false;
            _emergency = false;
            _endpoint = "";
            _loggedState = 0;
            _loggedVideo = 0;
            _lastTracedRc = "";
            _stateAtUtc = DateTime.MinValue;
            _telemetry.Clear();
        }

        try
        {
            session?.Cancel();
        }
        catch
        {
            // Already cancelled.
        }

        StopVideo();
        CloseSockets();
        try
        {
            session?.Dispose();
        }
        catch
        {
            // Already disposed.
        }
    }

    private void CloseSockets()
    {
        UdpClient? command;
        UdpClient? state;
        lock (_gate)
        {
            command = _command;
            state = _state;
            _command = null;
            _state = null;
            _pending?.TrySetCanceled();
            _pending = null;
        }

        try { command?.Close(); } catch { /* already closed */ }
        try { state?.Close(); } catch { /* already closed */ }
    }

    private void Log(string text) => Logged?.Invoke(text);

    private void Publish() => Changed?.Invoke();

    private static int? ReadCm(Dictionary<string, string> map, string key)
    {
        if (!map.TryGetValue(key, out var text))
            return null;
        if (!double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var number))
            return null;
        return (int)Math.Round(number);
    }
}
