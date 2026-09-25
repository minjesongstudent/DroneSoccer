using System.Collections.Concurrent;
using System.IO;
using System.Threading;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace DroneSoccerWPF;

/// <summary>
/// Writes the live JPEG frames to an H.264 MP4 on a side thread.
/// A full queue drops frames so recording cannot stall the live picture.
/// </summary>
public sealed class FlightRecorder : IDisposable
{
    public static string DefaultFolder =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyVideos), "DroneSoccer");

    private readonly object _gate = new();
    private BlockingCollection<byte[]>? _frames;
    private Thread? _thread;
    private string? _file;
    private int _recording;
    private int _dropped;
    private int _written;
    private string? _error;

    public string? LastError
    {
        get
        {
            lock (_gate)
                return _error;
        }
    }

    public bool IsRecording => Volatile.Read(ref _recording) == 1;

    public string? CurrentFile
    {
        get
        {
            lock (_gate)
                return _file;
        }
    }

    public string Start(string folder)
    {
        Stop();
        Directory.CreateDirectory(folder);
        var file = Path.Combine(folder, $"tello-{DateTime.Now:yyyyMMdd-HHmmss}.mp4");
        var frames = new BlockingCollection<byte[]>(60);
        lock (_gate)
        {
            _file = file;
            _frames = frames;
            _dropped = 0;
            _written = 0;
            _error = null;
        }

        Volatile.Write(ref _recording, 1);
        var thread = new Thread(() => WriteLoop(frames, file)) { IsBackground = true, Name = "video-record" };
        _thread = thread;
        thread.Start();
        return file;
    }

    public void Push(byte[] jpeg)
    {
        if (Volatile.Read(ref _recording) == 0 || jpeg.Length == 0)
            return;

        BlockingCollection<byte[]>? frames;
        lock (_gate)
            frames = _frames;
        if (frames is null)
            return;

        try
        {
            if (!frames.TryAdd(jpeg) && Interlocked.Exchange(ref _dropped, 1) == 0)
                return;
        }
        catch (InvalidOperationException)
        {
            // Stop already closed the queue.
        }
    }

    public string? Stop()
    {
        if (Interlocked.Exchange(ref _recording, 0) == 0)
            return null;

        BlockingCollection<byte[]>? frames;
        Thread? thread;
        string? file;
        lock (_gate)
        {
            frames = _frames;
            thread = _thread;
            file = _file;
            _frames = null;
            _thread = null;
        }

        try
        {
            frames?.CompleteAdding();
        }
        catch (InvalidOperationException)
        {
            // Already completed.
        }

        thread?.Join(TimeSpan.FromSeconds(5));
        frames?.Dispose();

        int written;
        lock (_gate)
        {
            written = _written;
            _file = null;
        }

        if (written <= 0)
        {
            TryDelete(file);
            return null;
        }

        return file;
    }

    public void Dispose() => Stop();

    private void WriteLoop(BlockingCollection<byte[]> frames, string file)
    {
        H264Mp4Writer? writer = null;
        string? error = null;
        var sawFrame = false;
        try
        {
            foreach (var jpeg in frames.GetConsumingEnumerable())
            {
                sawFrame = true;
                if (!TryDecode(jpeg, out var pixels, out var width, out var height, out var stride, out var step))
                    continue;

                if (writer is null)
                {
                    var evenWidth = width & ~1;
                    var evenHeight = height & ~1;
                    if (evenWidth < 2 || evenHeight < 2)
                        continue;

                    writer = new H264Mp4Writer();
                    writer.Open(file, evenWidth, evenHeight, 30);
                }
                else if (width < writer.Width || height < writer.Height)
                {
                    continue;
                }

                writer.WriteBgra(pixels, stride, step);
                lock (_gate)
                    _written++;
            }

            if (writer is null && sawFrame)
                error = "Could not decode a video frame.";
        }
        catch (Exception ex)
        {
            error = ex.Message;
        }
        finally
        {
            try
            {
                writer?.Dispose();
            }
            catch (Exception ex)
            {
                error ??= ex.Message;
                lock (_gate)
                    _written = 0;
            }
        }

        if (error is not null)
        {
            lock (_gate)
                _error = error;
        }
    }

    private static bool TryDecode(byte[] jpeg, out byte[] pixels, out int width, out int height, out int stride, out int step)
    {
        pixels = Array.Empty<byte>();
        width = 0;
        height = 0;
        stride = 0;
        step = 0;
        try
        {
            using var stream = new MemoryStream(jpeg, writable: false);
            BitmapSource frame = BitmapDecoder.Create(
                stream,
                BitmapCreateOptions.PreservePixelFormat,
                BitmapCacheOption.OnLoad).Frames[0];
            if (frame.Format == PixelFormats.Bgra32 || frame.Format == PixelFormats.Pbgra32)
                step = 4;
            else if (frame.Format == PixelFormats.Bgr24 || frame.Format == PixelFormats.Bgr32)
                step = frame.Format.BitsPerPixel / 8;
            else
            {
                frame = new FormatConvertedBitmap(frame, PixelFormats.Bgra32, null, 0);
                step = 4;
            }

            width = frame.PixelWidth;
            height = frame.PixelHeight;
            if (width < 2 || height < 2)
                return false;

            stride = width * step;
            pixels = new byte[stride * height];
            frame.CopyPixels(pixels, stride, 0);
            return true;
        }
        catch (Exception)
        {
            return false;
        }
    }

    private static void TryDelete(string? file)
    {
        if (string.IsNullOrWhiteSpace(file))
            return;
        try
        {
            if (File.Exists(file))
                File.Delete(file);
        }
        catch (IOException)
        {
            // The empty file can stay.
        }
    }
}
