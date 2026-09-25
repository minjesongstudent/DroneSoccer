using System.Runtime.InteropServices;

namespace DroneSoccerWPF;

/// <summary>
/// Writes BGRA frames to an H.264 MP4 that the Windows video apps can play.
/// Encoding uses Media Foundation, not OpenCV, so it does not share native state with the live decoder.
/// </summary>
public sealed class H264Mp4Writer : IDisposable
{
    const int MfVersion = 0x00020070;
    const int CoInitMultiThreaded = 0x0;
    const int RpcEChangedMode = unchecked((int)0x80010106);

    static readonly Guid MfMajorType = new("48eba18e-f8c9-4687-bf11-0a74c9f96a8f");
    static readonly Guid MfSubtype = new("f7e34c9a-42e8-4714-b74b-cb29d72c35e5");
    static readonly Guid MfAvgBitrate = new("20332624-fb0d-4d9e-bd0d-cbf6786c102e");
    static readonly Guid MfInterlace = new("e2724bb8-e676-4806-b4b2-a8d6efb44ccd");
    static readonly Guid MfFrameSize = new("1652c33d-d6b2-4012-b834-72030849a37d");
    static readonly Guid MfFrameRate = new("c459a2e8-3d2c-4e44-b132-fee5156c7bb0");
    static readonly Guid MfPixelAspect = new("c6376a1e-8d0a-4027-be45-6d9a0ad39bb6");
    static readonly Guid MfDefaultStride = new("644b4e48-1e02-4516-b0eb-c01ca9d49ac6");
    static readonly Guid MfAllSamplesIndependent = new("c9173739-5e56-461c-b713-46fb995cb95f");
    static readonly Guid MfFixedSizeSamples = new("b8ebefaf-b718-4e04-b0a9-116775e3321b");
    static readonly Guid MfSampleSize = new("dad3ab78-1990-408b-bce2-eba673dacc10");
    static readonly Guid MfHardwareTransforms = new("a634a91c-822b-41b9-a494-4de4643612b0");
    static readonly Guid MediaTypeVideo = new("73646976-0000-0010-8000-00AA00389B71");
    static readonly Guid VideoFormatH264 = new("34363248-0000-0010-8000-00AA00389B71");
    static readonly Guid VideoFormatNv12 = new("3231564E-0000-0010-8000-00AA00389B71");

    static int _mfStartupCount;

    readonly object _gate = new();
    ISinkWriter? _writer;
    byte[]? _nv12;
    int _width;
    int _height;
    int _fps;
    int _stream;
    long _frame;
    bool _comStarted;
    bool _finished;

    public int Width => _width;
    public int Height => _height;

    public void Open(string path, int width, int height, int framesPerSecond)
    {
        if (width <= 0 || height <= 0 || (width & 1) != 0 || (height & 1) != 0)
            throw new InvalidOperationException("H.264 needs a positive even frame size.");
        if (framesPerSecond <= 0)
            throw new InvalidOperationException("H.264 needs a positive frame rate.");

        EnsureMediaFoundation();
        var hr = CoInitializeEx(IntPtr.Zero, CoInitMultiThreaded);
        if (hr == RpcEChangedMode)
            throw new InvalidOperationException("The recording thread could not start the video encoder.");
        if (hr < 0)
            throw new InvalidOperationException($"Could not start the video encoder (0x{hr:X8}).");
        _comStarted = true;

        IMfAttributes? writerAttrs = null;
        IMfAttributes? output = null;
        IMfAttributes? input = null;
        try
        {
            Check(MFCreateAttributes(out writerAttrs, 1), "Could not configure the video encoder");
            var hardware = MfHardwareTransforms;
            Check(writerAttrs.SetUInt32(ref hardware, 0), "Could not configure the video encoder");
            Check(MFCreateSinkWriterFromURL(path, IntPtr.Zero, writerAttrs, out var writer), "Could not create the MP4 file");
            _writer = writer;

            output = CreateType();
            SetGuid(output, MfMajorType, MediaTypeVideo);
            SetGuid(output, MfSubtype, VideoFormatH264);
            SetUInt32(output, MfAvgBitrate, Bitrate(width, height, framesPerSecond));
            SetUInt32(output, MfInterlace, 2);
            SetUInt64(output, MfFrameSize, Pack((uint)width, (uint)height));
            SetUInt64(output, MfFrameRate, Pack((uint)framesPerSecond, 1));
            SetUInt64(output, MfPixelAspect, Pack(1, 1));
            Check(_writer.AddStream(output, out _stream), "Could not add the H.264 stream");

            var nv12Bytes = width * height * 3 / 2;
            input = CreateType();
            SetGuid(input, MfMajorType, MediaTypeVideo);
            SetGuid(input, MfSubtype, VideoFormatNv12);
            SetUInt32(input, MfInterlace, 2);
            SetUInt64(input, MfFrameSize, Pack((uint)width, (uint)height));
            SetUInt64(input, MfFrameRate, Pack((uint)framesPerSecond, 1));
            SetUInt64(input, MfPixelAspect, Pack(1, 1));
            SetUInt32(input, MfDefaultStride, width);
            SetUInt32(input, MfAllSamplesIndependent, 1);
            SetUInt32(input, MfFixedSizeSamples, 1);
            SetUInt32(input, MfSampleSize, nv12Bytes);
            Check(_writer.SetInputMediaType(_stream, input, null), "Could not set the H.264 input format");
            Check(_writer.BeginWriting(), "Could not start the H.264 encoder");
            _nv12 = new byte[nv12Bytes];
            _width = width;
            _height = height;
            _fps = framesPerSecond;
        }
        catch
        {
            Release(_writer);
            _writer = null;
            if (_comStarted)
            {
                CoUninitialize();
                _comStarted = false;
            }

            throw;
        }
        finally
        {
            Release(writerAttrs);
            Release(output);
            Release(input);
        }
    }

    public void WriteBgra(byte[] pixels, int stride, int pixelStep)
    {
        if (_writer is null || _nv12 is null)
            throw new InvalidOperationException("The MP4 file is not open.");
        if (pixelStep != 3 && pixelStep != 4)
            throw new InvalidOperationException("The frame is not BGR or BGRA.");
        if (stride < _width * pixelStep || pixels.Length < stride * _height)
            throw new InvalidOperationException("The frame is smaller than the recording size.");

        ConvertToNv12(pixels, stride, pixelStep, _width, _height, _nv12);
        WriteNv12(_nv12);
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_finished)
                return;
            _finished = true;
        }

        try
        {
            if (_writer is not null && _frame > 0)
                Check(_writer.FinalizeWriter(), "Could not finish the MP4 file");
        }
        finally
        {
            Release(_writer);
            _writer = null;
            if (_comStarted)
            {
                CoUninitialize();
                _comStarted = false;
            }
        }
    }

    void WriteNv12(byte[] nv12)
    {
        IMediaBuffer? buffer = null;
        IMfSample? sample = null;
        var locked = false;
        try
        {
            Check(MFCreateMemoryBuffer(nv12.Length, out buffer), "Could not allocate a video frame");
            Check(buffer.Lock(out var dest, out _, out _), "Could not lock a video frame");
            locked = true;
            Marshal.Copy(nv12, 0, dest, nv12.Length);
            buffer.Unlock();
            locked = false;
            Check(buffer.SetCurrentLength(nv12.Length), "Could not set the video frame size");

            Check(MFCreateSample(out sample), "Could not create a video sample");
            Check(sample.AddBuffer(buffer), "Could not attach a video frame");
            var time = _frame * 10_000_000L / _fps;
            var duration = 10_000_000L / _fps;
            Check(sample.SetSampleTime(time), "Could not set the video timestamp");
            Check(sample.SetSampleDuration(duration), "Could not set the video duration");
            Check(_writer!.WriteSample(_stream, sample), "Could not write a video frame");
            _frame++;
        }
        finally
        {
            if (locked)
                buffer?.Unlock();
            Release(sample);
            Release(buffer);
        }
    }

    static void ConvertToNv12(byte[] pixels, int stride, int step, int width, int height, byte[] nv12)
    {
        var ySize = width * height;
        for (var y = 0; y < height; y++)
        {
            var row = y * stride;
            var yRow = y * width;
            for (var x = 0; x < width; x++)
            {
                var i = row + x * step;
                var b = pixels[i];
                var g = pixels[i + 1];
                var r = pixels[i + 2];
                nv12[yRow + x] = Clamp(((66 * r + 129 * g + 25 * b + 128) >> 8) + 16);
            }
        }

        var uv = ySize;
        for (var y = 0; y < height; y += 2)
        {
            var row = y * stride;
            for (var x = 0; x < width; x += 2)
            {
                var i = row + x * step;
                var b = pixels[i];
                var g = pixels[i + 1];
                var r = pixels[i + 2];
                nv12[uv++] = Clamp(((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128);
                nv12[uv++] = Clamp(((112 * r - 94 * g - 18 * b + 128) >> 8) + 128);
            }
        }
    }

    static byte Clamp(int value) => (byte)(value < 0 ? 0 : value > 255 ? 255 : value);

    static int Bitrate(int width, int height, int fps)
    {
        var bits = (long)width * height * fps / 4;
        if (bits < 1_000_000)
            return 1_000_000;
        if (bits > 8_000_000)
            return 8_000_000;
        return (int)bits;
    }

    static long Pack(uint high, uint low) => ((long)high << 32) | low;

    static IMfAttributes CreateType()
    {
        Check(MFCreateMediaType(out var type), "Could not create a media type");
        return type;
    }

    static void SetGuid(IMfAttributes attrs, Guid key, Guid value)
    {
        Check(attrs.SetGuid(ref key, ref value), "Could not set a media type field");
    }

    static void SetUInt32(IMfAttributes attrs, Guid key, int value)
    {
        Check(attrs.SetUInt32(ref key, value), "Could not set a media type field");
    }

    static void SetUInt64(IMfAttributes attrs, Guid key, long value)
    {
        Check(attrs.SetUInt64(ref key, value), "Could not set a media type field");
    }

    static void EnsureMediaFoundation()
    {
        if (Interlocked.Increment(ref _mfStartupCount) != 1)
            return;
        var hr = MFStartup(MfVersion, 0);
        if (hr < 0)
        {
            Interlocked.Decrement(ref _mfStartupCount);
            throw new InvalidOperationException($"Could not start the video encoder (0x{hr:X8}).");
        }
    }

    static void Check(int hr, string action)
    {
        if (hr < 0)
            throw new InvalidOperationException($"{action} (0x{hr:X8}).");
    }

    static void Release(object? com)
    {
        if (com is not null && Marshal.IsComObject(com))
            Marshal.ReleaseComObject(com);
    }

    [DllImport("ole32.dll", ExactSpelling = true)]
    static extern int CoInitializeEx(IntPtr reserved, int coInit);

    [DllImport("ole32.dll", ExactSpelling = true)]
    static extern void CoUninitialize();

    [DllImport("mfplat.dll", ExactSpelling = true)]
    static extern int MFStartup(int version, int flags);

    [DllImport("mfplat.dll", ExactSpelling = true)]
    static extern int MFCreateAttributes(out IMfAttributes attributes, uint initialSize);

    [DllImport("mfplat.dll", ExactSpelling = true)]
    static extern int MFCreateMediaType(out IMfAttributes mediaType);

    [DllImport("mfplat.dll", ExactSpelling = true)]
    static extern int MFCreateSample(out IMfSample sample);

    [DllImport("mfplat.dll", ExactSpelling = true)]
    static extern int MFCreateMemoryBuffer(int maxLength, out IMediaBuffer buffer);

    [DllImport("mfreadwrite.dll", ExactSpelling = true)]
    static extern int MFCreateSinkWriterFromURL(
        [MarshalAs(UnmanagedType.LPWStr)] string url,
        IntPtr byteStream,
        IMfAttributes attributes,
        out ISinkWriter writer);

    [ComImport]
    [Guid("2cd2d921-c447-44a7-a13c-4adabfc247e3")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IMfAttributes
    {
        void GetItem(ref Guid key, IntPtr value);
        void GetItemType(ref Guid key, out int type);
        void CompareItem(ref Guid key, IntPtr value, out int result);
        void Compare(IMfAttributes theirs, int matchType, out int result);
        void GetUInt32Value(ref Guid key, out int value);
        void GetUInt64Value(ref Guid key, out long value);
        void GetDouble(ref Guid key, out double value);
        void GetGuidValue(ref Guid key, out Guid value);
        void GetStringLength(ref Guid key, out int length);
        void GetString(ref Guid key, IntPtr value, int bufSize, IntPtr length);
        void GetAllocatedString(ref Guid key, out IntPtr value, out int length);
        void GetBlobSize(ref Guid key, out int size);
        void GetBlob(ref Guid key, IntPtr buf, int bufSize, IntPtr size);
        void GetAllocatedBlob(ref Guid key, out IntPtr buf, out int size);
        void GetUnknown(ref Guid key, ref Guid iid, out IntPtr unknown);
        void SetItem(ref Guid key, IntPtr value);
        void DeleteItem(ref Guid key);
        void DeleteAllItems();
        [PreserveSig] int SetUInt32(ref Guid key, int value);
        [PreserveSig] int SetUInt64(ref Guid key, long value);
        void SetDouble(ref Guid key, double value);
        [PreserveSig] int SetGuid(ref Guid key, ref Guid value);
        void SetString(ref Guid key, [MarshalAs(UnmanagedType.LPWStr)] string value);
        void SetBlob(ref Guid key, IntPtr buf, int size);
        void SetUnknown(ref Guid key, IntPtr unknown);
        void LockStore();
        void UnlockStore();
        void GetCount(out int count);
        void GetItemByIndex(int index, out Guid key, IntPtr value);
        void CopyAllItems(IMfAttributes dest);
    }

    [ComImport]
    [Guid("045FA593-8799-42b8-BC8D-8968C6453507")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IMediaBuffer
    {
        [PreserveSig] int Lock(out IntPtr buffer, out int maxLength, out int currentLength);
        [PreserveSig] int Unlock();
        void GetCurrentLength(out int length);
        [PreserveSig] int SetCurrentLength(int length);
        void GetMaxLength(out int length);
    }

    // IMFSample starts with every IMFAttributes method, then the sample methods.
    [ComImport]
    [Guid("c40a00f2-b93a-4d80-ae8c-5a1c634f58e4")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IMfSample
    {
        void GetItem(ref Guid key, IntPtr value);
        void GetItemType(ref Guid key, out int type);
        void CompareItem(ref Guid key, IntPtr value, out int result);
        void Compare(IMfAttributes theirs, int matchType, out int result);
        void GetUInt32Value(ref Guid key, out int value);
        void GetUInt64Value(ref Guid key, out long value);
        void GetDouble(ref Guid key, out double value);
        void GetGuidValue(ref Guid key, out Guid value);
        void GetStringLength(ref Guid key, out int length);
        void GetString(ref Guid key, IntPtr value, int bufSize, IntPtr length);
        void GetAllocatedString(ref Guid key, out IntPtr value, out int length);
        void GetBlobSize(ref Guid key, out int size);
        void GetBlob(ref Guid key, IntPtr buf, int bufSize, IntPtr size);
        void GetAllocatedBlob(ref Guid key, out IntPtr buf, out int size);
        void GetUnknown(ref Guid key, ref Guid iid, out IntPtr unknown);
        void SetItem(ref Guid key, IntPtr value);
        void DeleteItem(ref Guid key);
        void DeleteAllItems();
        void SetUInt32(ref Guid key, int value);
        void SetUInt64(ref Guid key, long value);
        void SetDouble(ref Guid key, double value);
        void SetGuid(ref Guid key, ref Guid value);
        void SetString(ref Guid key, [MarshalAs(UnmanagedType.LPWStr)] string value);
        void SetBlob(ref Guid key, IntPtr buf, int size);
        void SetUnknown(ref Guid key, IntPtr unknown);
        void LockStore();
        void UnlockStore();
        void GetCount(out int count);
        void GetItemByIndex(int index, out Guid key, IntPtr value);
        void CopyAllItems(IMfAttributes dest);
        void GetSampleFlags(out int flags);
        void SetSampleFlags(int flags);
        void GetSampleTime(out long time);
        [PreserveSig] int SetSampleTime(long time);
        void GetSampleDuration(out long duration);
        [PreserveSig] int SetSampleDuration(long duration);
        void GetBufferCount(out int count);
        void GetBufferByIndex(int index, out IMediaBuffer buffer);
        void ConvertToContiguousBuffer(out IMediaBuffer buffer);
        [PreserveSig] int AddBuffer(IMediaBuffer buffer);
    }

    [ComImport]
    [Guid("3137f1cd-fe5e-4805-a5d8-fb477448cb3d")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface ISinkWriter
    {
        [PreserveSig] int AddStream(IMfAttributes target, out int streamIndex);
        [PreserveSig] int SetInputMediaType(int streamIndex, IMfAttributes input, IMfAttributes? parameters);
        [PreserveSig] int BeginWriting();
        [PreserveSig] int WriteSample(int streamIndex, IMfSample sample);
        void SendStreamTick(int streamIndex, long timestamp);
        void PlaceMarker(int streamIndex, IntPtr context);
        void NotifyEndOfSegment(int streamIndex);
        void Flush(int streamIndex);
        [PreserveSig] int FinalizeWriter();
    }
}
