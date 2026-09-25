using System.IO;
using System.Net;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace DroneSoccerWPF;

public sealed class AppSettings
{
    public int MaxHeightCm { get; set; } = 150;
    public string DroneIp { get; set; } = "192.168.10.1";
    public int CommandPort { get; set; } = TelloLink.CommandPort;
    public string RecordFolder { get; set; } = "";
    public string PythonIp { get; set; } = PythonLink.DefaultHost;
    public int PythonControlPort { get; set; } = PythonLink.DefaultControlPort;
    public int PythonVideoPort { get; set; } = PythonLink.DefaultVideoPort;
}

public static class SettingsStore
{
    private static readonly JsonSerializerOptions WriteOptions = new() { WriteIndented = true };

    public static AppSettings Load(string path)
    {
        var settings = new AppSettings();
        if (!File.Exists(path))
            return settings;

        try
        {
            var node = JsonNode.Parse(File.ReadAllText(path)) as JsonObject;
            if (node is null)
                return settings;

            if (node["max_height_cm"] is JsonValue heightValue && heightValue.TryGetValue<int>(out var height))
                settings.MaxHeightCm = Math.Clamp(height, 50, 1000);
            if (node["drone_ip"] is JsonValue ipValue && ipValue.TryGetValue<string>(out var ip) && IsIpv4(ip))
                settings.DroneIp = ip.Trim();
            if (node["command_port"] is JsonValue portValue && portValue.TryGetValue<int>(out var port) && port is > 0 and <= 65535)
                settings.CommandPort = port;
            if (node["record_folder"] is JsonValue folderValue && folderValue.TryGetValue<string>(out var folder) && !string.IsNullOrWhiteSpace(folder))
                settings.RecordFolder = folder.Trim();
            if (node["python_ip"] is JsonValue pythonIpValue && pythonIpValue.TryGetValue<string>(out var pythonIp) && IsIpv4(pythonIp))
                settings.PythonIp = pythonIp.Trim();
            if (node["python_control_port"] is JsonValue pythonControlValue && pythonControlValue.TryGetValue<int>(out var pythonControl) && pythonControl is > 0 and <= 65535)
                settings.PythonControlPort = pythonControl;
            if (node["python_video_port"] is JsonValue pythonVideoValue && pythonVideoValue.TryGetValue<int>(out var pythonVideo) && pythonVideo is > 0 and <= 65535)
                settings.PythonVideoPort = pythonVideo;
        }
        catch (JsonException)
        {
            // Keep the defaults when the file is unreadable.
        }
        catch (IOException)
        {
            // Keep the defaults when the file cannot be read.
        }

        return settings;
    }

    public static void Save(string path, AppSettings settings)
    {
        JsonObject root;
        try
        {
            root = File.Exists(path)
                ? JsonNode.Parse(File.ReadAllText(path)) as JsonObject ?? new JsonObject()
                : new JsonObject();
        }
        catch (Exception ex) when (ex is JsonException or IOException)
        {
            root = new JsonObject();
        }

        root["max_height_cm"] = settings.MaxHeightCm;
        root["drone_ip"] = settings.DroneIp;
        root["command_port"] = settings.CommandPort;
        root["record_folder"] = settings.RecordFolder;
        root["python_ip"] = settings.PythonIp;
        root["python_control_port"] = settings.PythonControlPort;
        root["python_video_port"] = settings.PythonVideoPort;
        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);
        File.WriteAllText(path, root.ToJsonString(WriteOptions) + Environment.NewLine);
    }

    private static bool IsIpv4(string text)
    {
        return IPAddress.TryParse(text.Trim(), out var address) && address.AddressFamily == System.Net.Sockets.AddressFamily.InterNetwork;
    }
}
