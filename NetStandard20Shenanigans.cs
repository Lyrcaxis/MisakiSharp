namespace MisakiSharp;

using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;

/// <summary> Fills netstandard2.0's BCL gaps (Unity/VSIX/.NET Framework) with the exact same APIs newer TFMs ship built-in. </summary>
/// <remarks> Compiled only for the netstandard2.0 target -- on net8.0 the real BCL members bind instead, so behavior is identical across targets. </remarks>
internal static class NetStandard20Shenanigans {
    public static bool StartsWith(this string text, char c) => text.Length > 0 && text[0] == c;
    public static bool EndsWith(this string text, char c) => text.Length > 0 && text[text.Length - 1] == c;
    public static bool Contains(this string text, char c) => text.IndexOf(c) >= 0;
    public static string[] Split(this string text, char separator, StringSplitOptions options) => text.Split(new[] { separator }, options);
    public static string[] Split(this string text, char separator, int count) => text.Split(new[] { separator }, count);

    public static void Deconstruct<K, V>(this KeyValuePair<K, V> pair, out K key, out V value) { (key, value) = (pair.Key, pair.Value); }
    public static V GetValueOrDefault<K, V>(this IDictionary<K, V> dict, K key) => dict.TryGetValue(key, out var value) ? value : default;
    public static V GetValueOrDefault<K, V>(this IDictionary<K, V> dict, K key, V defaultValue) => dict.TryGetValue(key, out var value) ? value : defaultValue;
    public static HashSet<T> ToHashSet<T>(this IEnumerable<T> source) => new(source);
    public static IOrderedEnumerable<T> Order<T>(this IEnumerable<T> source) => source.OrderBy(x => x);
    public static T MaxBy<T, TKey>(this IEnumerable<T> source, Func<T, TKey> keySelector) => source.OrderByDescending(keySelector).FirstOrDefault();
    public static T Max<T>(this IEnumerable<T> source, IComparer<T> comparer) => source.OrderByDescending(x => x, comparer).FirstOrDefault();
    public static IEnumerable<(T1 First, T2 Second)> Zip<T1, T2>(this IEnumerable<T1> first, IEnumerable<T2> second) => first.Zip(second, (a, b) => (a, b));
    public static IEnumerable<T> Select<T>(this MatchCollection matches, Func<Match, T> selector) => matches.Cast<Match>().Select(selector);

    public static int Read7BitEncodedInt(this BinaryReader reader) {
        var (result, shift) = (0, 0);
        byte b;
        do { b = reader.ReadByte(); result |= (b & 0x7F) << shift; shift += 7; } while ((b & 0x80) != 0);
        return result;
    }
    public static int Read(this BinaryReader reader, Span<byte> buffer) {
        var temp = new byte[buffer.Length];
        var got = reader.Read(temp, 0, temp.Length);
        temp.AsSpan(0, got).CopyTo(buffer);
        return got;
    }
    public static int GetBytes(this Encoding encoding, string text, byte[] destination) => encoding.GetBytes(text, 0, text.Length, destination, 0);
}

internal static class NativeLibrary {
    public static IntPtr Load(string path) {
        var handle = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? LoadLibrary(path) : dlopen(path, 0x002 /* RTLD_NOW */);
        return handle != IntPtr.Zero ? handle : throw new DllNotFoundException($"Failed to load native library '{path}'.");
    }
    public static IntPtr GetExport(IntPtr handle, string name) {
        var export = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? GetProcAddress(handle, name) : dlsym(handle, name);
        return export != IntPtr.Zero ? export : throw new EntryPointNotFoundException($"Export '{name}' not found.");
    }

    [DllImport("kernel32", SetLastError = true, CharSet = CharSet.Ansi)] static extern IntPtr LoadLibrary(string path);
    [DllImport("kernel32", CharSet = CharSet.Ansi)] static extern IntPtr GetProcAddress(IntPtr module, string name);
    [DllImport("libdl", CharSet = CharSet.Ansi)] static extern IntPtr dlopen(string path, int flags);
    [DllImport("libdl", CharSet = CharSet.Ansi)] static extern IntPtr dlsym(IntPtr handle, string name);
}
