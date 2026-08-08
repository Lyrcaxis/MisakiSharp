namespace MisakiSharp;

using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;

/// <summary> Espeak-free Hindi: replays an espeak-ng 1.52.0 dump (positional word lexicon + number slot tables) instead of invoking espeak. </summary>
/// <remarks>
/// Plug straight into the espeak pipeline: <c>new EspeakG2P(HindiProvider.Phonemize)</c> -- no espeak binaries involved.
/// The dump captures espeak's quirks verbatim: flagged function words destress by clause position ('mein' keeps stress clause-finally,
/// 'ki' never does), numbers use distinct standalone/after-hundred/group forms (Indian ladder: sau/hazaar/lakh/crore/arab/kharab),
/// a danda attached to a word is silent while a standalone leading danda is verbalized, and Latin words carry their (en) switch flags.
/// Out-of-lexicon words are skipped (espeak parity is impossible without espeak); regenerate the dump via generators/generate_hi_native.py.
/// </remarks>
public static partial class HindiProvider {
    readonly record struct Forms(string Alone, string Nonfinal, string Final);

    static readonly Dictionary<string, Forms> lexicon = [];
    static readonly string[] num = new string[100], hundredFused = new string[10], tailAfterHundred = new string[100],
                             tailFinal = new string[100], group = new string[100], digitwise = new string[10];
    static readonly Dictionary<int, string> scales = [];

    static HindiProvider() {
        using var reader = new StreamReader(new GZipStream(typeof(HindiProvider).Assembly.GetManifestResourceStream("MisakiSharp.hi_espeak_lexicon.tsv.gz"), CompressionMode.Decompress), Encoding.UTF8);
        while (reader.ReadLine() is string line) {
            if (line.Length == 0 || line[0] == '#') { continue; }
            var cols = line.Split('\t');
            if (cols[0].StartsWith("@@N", StringComparison.Ordinal)) {
                var table = cols[0] switch { "@@N" => num, "@@NH" => hundredFused, "@@NT" => tailAfterHundred, "@@NF" => tailFinal, "@@NG" => group, "@@ND" => digitwise, _ => null };
                if (table is not null) { table[int.Parse(cols[1])] = cols[2]; } else { scales[int.Parse(cols[1])] = cols[2]; }
            } else { lexicon[cols[0]] = cols.Length == 2 ? new(cols[1], cols[1], cols[1]) : new(cols[1], cols[2], cols[3]); }
        }
    }

    /// <summary> The chunk provider: whitespace tokens through the dump, positioned within the clause, numbers composed on the fly. Out-of-lexicon words go silent. </summary>
    public static string Phonemize(string chunk) => Phonemize(chunk, out _);

    /// <summary> Hybrid provider: native replay, except chunks containing out-of-lexicon words route to the given espeak provider whole. </summary>
    /// <remarks> Keeps espeak required but rarely spawned (~70% fewer calls on wiki text) with zero correctness loss -- the stepping stone until a grapheme fallback lands. </remarks>
    public static Func<string, string> WithFallback(Func<string, string> espeakProvider)
        => chunk => Phonemize(chunk, out bool hadUnknownWords) is var native && hadUnknownWords ? espeakProvider(chunk) : native;

    static string Phonemize(string chunk, out bool hadUnknownWords) {
        var spoken = chunk.Split(default(char[]), StringSplitOptions.RemoveEmptyEntries)
            .Select(t => (t.Trim('।', '॥') is { Length: > 0 } core ? core : t).Replace("‌", ""))
            .Where(t => t.Length > 0).ToList();
        int wordCount = spoken.Count(t => t is not ("।" or "॥"));
        var (rendered, saidSomething, wi) = (new List<string>(), false, -1);
        hadUnknownWords = false;
        foreach (var token in spoken) {
            if (token is "।" or "॥") {
                if (!saidSomething) { rendered.Add(lexicon[token].Alone); saidSomething = true; }
            } else if (++wi >= 0 && NumberToken().IsMatch(token)) { rendered.Add(NumberPhonemes(token)); saidSomething = true; }
            else if (lexicon.TryGetValue(token, out var forms)) {
                rendered.Add(wordCount == 1 ? forms.Alone : wi == wordCount - 1 ? forms.Final : forms.Nonfinal);
                saidSomething = true;
            } else { hadUnknownWords = true; }
        }
        return string.Join(' ', rendered.Where(p => p.Length > 0));
    }

    static string FinalGroup(int n, bool hasHigherGroups) {
        if (n < 100) { return hasHigherGroups ? tailFinal[n] : num[n]; }
        return hundredFused[n / 100] + (n % 100 > 0 ? $" {tailAfterHundred[n % 100]}" : "");
    }

    static string NumberPhonemes(string digits) {
        digits = string.Concat(digits.Select(c => c is >= '०' and <= '९' ? (char)(c - '०' + '0') : c));
        if (digits[0] == '0' || digits.Length > 13) {
            if (digits.Length == 1) { return digitwise[digits[0] - '0']; }
            if (digits.Length == 2) { return $"{digitwise[digits[0] - '0']} {digitwise[digits[1] - '0']}"; }
            var joined = string.Concat(digits.Select(d => digitwise[d - '0']));
            return digits.Length != 3 ? joined : $"{digitwise[digits[0] - '0']}{digitwise[digits[1] - '0']} {digitwise[digits[2] - '0']}";
        }
        long n = long.Parse(digits);
        var parts = new List<string>();
        foreach (var (power, divisor) in ((int, long)[])[(11, 100_000_000_000), (9, 1_000_000_000), (7, 10_000_000), (5, 100_000), (3, 1000)]) {
            int g = (int)(n / divisor % 100);
            if (g > 0) { parts.Add($"{group[g]} {scales[power]}"); }
        }
        if (n % 1000 > 0) { parts.Add(FinalGroup((int)(n % 1000), n >= 1000)); }
        return string.Join(' ', parts);
    }

    [GeneratedRegex(@"^[०-९0-9]+$")] private static partial Regex NumberToken();
}
