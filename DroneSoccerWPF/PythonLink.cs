using System.Buffers.Binary;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace DroneSoccerWPF;

/// <summary>
/// Link to the Python vision and policy process. Video frames go out on 8766.
/// Results come back as JSON lines on 8765. This socket never talks to the aircraft.
/// </summary>
public sealed class PythonLink : IDisposable
{
    public const string DefaultHost = "127.0.0.1";
    public const int DefaultControlPort = 8765;
    public const int DefaultVideoPort = 8766;

    private readonly SemaphoreSlim _send = new(1, 1);
    private readonly SemaphoreSlim _videoSend = new(1, 1);
    private TcpClient? _control;
    private TcpClient? _video;
    private NetworkStream? _controlStream;
    private NetworkStream? _videoStream;
    private CancellationTokenSource? _sessionCts;
    private int _open;
    private int _frameBusy;

    public event Action<JsonElement>? MessageReceived;
    public event Action<bool>? ConnectionChanged;

    public bool IsControlConnected => Volatile.Read(ref _open) == 1;

    public string Endpoint { get; private set; } = "";

    public async Task ConnectAsync(IPAddress address, int controlPort, int videoPort, CancellationToken cancellationToken)
    {
        if (IsControlConnected)
            return;
        if (controlPort == videoPort)
            throw new InvalidOperationException("The Python result port and video port must be different.");

        TcpClient? control = null;
        TcpClient? video = null;
        try
        {
            control = new TcpClient();
            await ConnectWithTimeout(control, address, controlPort, cancellationToken).ConfigureAwait(false);

            video = new TcpClient();
            await ConnectWithTimeout(video, address, videoPort, cancellationToken).ConfigureAwait(false);

            _control = control;
            _video = video;
            _controlStream = control.GetStream();
            _videoStream = video.GetStream();
            _sessionCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            Endpoint = $"{address}:{controlPort}   video {videoPort}";
            Volatile.Write(ref _open, 1);

            var token = _sessionCts.Token;
            _ = Task.Run(() => ReadControlAsync(token), token);
            ConnectionChanged?.Invoke(true);
        }
        catch
        {
            control?.Close();
            video?.Close();
            Endpoint = "";
            throw;
        }
    }

    public void Disconnect()
    {
        try
        {
            _sessionCts?.Cancel();
        }
        catch
        {
            // Closing the socket unblocks the reader.
        }

        Drop();
    }

    private static async Task ConnectWithTimeout(TcpClient client, IPAddress address, int port, CancellationToken cancellationToken)
    {
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(TimeSpan.FromSeconds(5));
        try
        {
            await client.ConnectAsync(address, port, timeout.Token).ConfigureAwait(false);
            client.NoDelay = true;
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            throw new TimeoutException($"No Python server answered at {address}:{port}.");
        }
    }

    public async Task SendAsync(object payload)
    {
        var stream = _controlStream;
        if (stream is null || !IsControlConnected)
            return;

        var body = JsonSerializer.SerializeToUtf8Bytes(payload);
        var packet = new byte[body.Length + 1];
        body.CopyTo(packet, 0);
        packet[^1] = (byte)'\n';

        await _send.WaitAsync().ConfigureAwait(false);
        try
        {
            await stream.WriteAsync(packet).ConfigureAwait(false);
            await stream.FlushAsync().ConfigureAwait(false);
        }
        catch
        {
            Drop();
        }
        finally
        {
            _send.Release();
        }
    }

    public void SendFrame(byte[] jpeg)
    {
        if (!IsControlConnected || jpeg.Length == 0 || jpeg.Length > 2_000_000)
            return;
        if (Interlocked.CompareExchange(ref _frameBusy, 1, 0) != 0)
            return;

        _ = Task.Run(async () =>
        {
            try
            {
                await WriteFrameAsync(jpeg).ConfigureAwait(false);
            }
            finally
            {
                Interlocked.Exchange(ref _frameBusy, 0);
            }
        });
    }

    public void Dispose()
    {
        try
        {
            _sessionCts?.Cancel();
        }
        catch
        {
            // Closing the socket unblocks the reader.
        }

        Drop();
        _send.Dispose();
        _videoSend.Dispose();
    }

    private async Task WriteFrameAsync(byte[] jpeg)
    {
        var stream = _videoStream;
        if (stream is null || !IsControlConnected)
            return;

        var header = new byte[4];
        BinaryPrimitives.WriteInt32BigEndian(header, jpeg.Length);

        await _videoSend.WaitAsync().ConfigureAwait(false);
        try
        {
            await stream.WriteAsync(header).ConfigureAwait(false);
            await stream.WriteAsync(jpeg).ConfigureAwait(false);
        }
        catch
        {
            Drop();
        }
        finally
        {
            _videoSend.Release();
        }
    }

    private async Task ReadControlAsync(CancellationToken cancellationToken)
    {
        try
        {
            var stream = _controlStream;
            if (stream is null)
                return;

            using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: false, bufferSize: 1024, leaveOpen: true);
            while (!cancellationToken.IsCancellationRequested)
            {
                var line = await reader.ReadLineAsync(cancellationToken).ConfigureAwait(false);
                if (line is null)
                    break;
                if (line.Length == 0)
                    continue;

                JsonElement message;
                try
                {
                    using var document = JsonDocument.Parse(line);
                    message = document.RootElement.Clone();
                }
                catch (JsonException)
                {
                    continue;
                }

                MessageReceived?.Invoke(message);
            }
        }
        catch (OperationCanceledException)
        {
            // Shutting down.
        }
        catch (IOException)
        {
            // Python closed the socket.
        }
        finally
        {
            Drop();
        }
    }

    private void Drop()
    {
        if (Interlocked.Exchange(ref _open, 0) == 0)
            return;

        try { _control?.Close(); } catch { /* already closed */ }
        try { _video?.Close(); } catch { /* already closed */ }
        _controlStream = null;
        _videoStream = null;
        Endpoint = "";
        ConnectionChanged?.Invoke(false);
    }
}
