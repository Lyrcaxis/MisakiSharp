namespace MisakiSharp;

using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;

/// <summary> Native port of spacy's English tokenizer (en_core_web_sm defaults), reproducing its exact token/whitespace stream without python. </summary>
/// <remarks>
/// spacy tokenizes per whitespace-separated chunk -- special cases (contractions, abbreviations, emoticons) match whole, otherwise prefixes/suffixes get
/// peeled off in a loop, the remainder is kept intact if it looks like a URL, or gets split on infixes (hyphens, ellipses, letter-dot-letter and friends).
/// A final pass re-merges special cases that affix peeling tore apart (e.g. ":)" glued to a word). Tables come from generate_en_tokenizer.py, with the
/// affix regexes pre-translated to .NET syntax (astral ranges become surrogate pair alternations, which python's re expresses directly).
/// </remarks>
public static class EnglishTokenizer {
    /// <summary> Optional path to en_tokenizer.tsv.gz on disk, for running without the embedded resource wired up. </summary>
    public static string TablePath { get; set; }

    static Regex prefixSearch, suffixSearch, infixFinditer, urlMatch;
    static readonly Dictionary<string, string[]> specialRules = [];
    static readonly Lazy<Dictionary<string, List<string[]>>> specialMatcher = new(LoadTables);

    /// <summary> Tokenizes text exactly like spacy's `[(t.text, t.whitespace_) for t in nlp.tokenizer(text)]` -- concatenating Text+Whitespace reproduces the input. </summary>
    public static List<(string Text, string Whitespace)> Tokenize(string text) {
        _ = specialMatcher.Value;
        var tokens = TokenizeAffixes(text, withSpecials: true);
        ApplySpecialCases(tokens);
        return tokens.Select(t => (t.Text, t.Spacy ? " " : "")).ToList();
    }

    static StreamReader OpenTable()
        => new(new GZipStream(TablePath is string path ? File.OpenRead(path) : typeof(EnglishTokenizer).Assembly.GetManifestResourceStream("MisakiSharp.en_tokenizer.tsv.gz"), CompressionMode.Decompress), Encoding.UTF8);

    /// <summary> Loads the affix/url regexes and the special-case rules, then precomputes the re-merge patterns (rules whose text affix-splitting would break). </summary>
    static Dictionary<string, List<string[]>> LoadTables() {
        using (var reader = OpenTable()) {
            while (reader.ReadLine() is string line) {
                var (key, value) = (line[..line.IndexOf('\t')], line[(line.IndexOf('\t') + 1)..]);
                if (key == "P") { prefixSearch = new Regex(value, RegexOptions.Compiled | RegexOptions.CultureInvariant); }
                else if (key == "S") { suffixSearch = new Regex(value, RegexOptions.Compiled | RegexOptions.CultureInvariant); }
                else if (key == "I") { infixFinditer = new Regex(value, RegexOptions.Compiled | RegexOptions.CultureInvariant); }
                else if (key == "U") { urlMatch = new Regex(value, RegexOptions.Compiled | RegexOptions.CultureInvariant); }
                else { specialRules[Unescape(key[1..])] = value.Split('\t').Select(Unescape).ToArray(); }
            }
        }
        var matcher = new Dictionary<string, List<string[]>>();
        foreach (var chunk in specialRules.Keys) {
            if (FindPrefix(chunk) == 0 && FindSuffix(chunk) == 0 && !infixFinditer.IsMatch(chunk) && !chunk.Contains(' ')) { continue; }
            var orths = TokenizeAffixes(chunk, withSpecials: false).Select(t => t.Text).ToArray();
            if (!matcher.TryGetValue(orths[0], out var patterns)) { matcher[orths[0]] = patterns = []; }
            patterns.Add(orths);
        }
        return matcher;
    }

    static string Unescape(string field) {
        if (!field.Contains('\\')) { return field; }
        var sb = new StringBuilder();
        for (int i = 0; i < field.Length; i++) {
            if (field[i] != '\\') { sb.Append(field[i]); continue; }
            sb.Append(field[++i] switch { 't' => '\t', 'n' => '\n', 'r' => '\r', var c => c });
        }
        return sb.ToString();
    }

    /// <summary> spacy's whitespace chunking -- single spaces become the previous token's whitespace flag, any other whitespace run becomes a token itself. </summary>
    static List<(string Text, bool Spacy)> TokenizeAffixes(string text, bool withSpecials) {
        var tokens = new List<(string Text, bool Spacy)>();
        if (text.Length == 0) { return tokens; }
        var (start, inWhitespace) = (0, IsSpace(text[0]));
        for (int i = 0; i < text.Length; i++) {
            if (IsSpace(text[i]) == inWhitespace) { continue; }
            if (start < i) { TokenizeChunk(tokens, text[start..i], withSpecials); }
            if (text[i] == ' ') { tokens[^1] = (tokens[^1].Text, true); start = i + 1; }
            else { start = i; }
            inWhitespace = !inWhitespace;
        }
        if (start < text.Length) { TokenizeChunk(tokens, text[start..], withSpecials); }
        return tokens;
    }

    static bool IsSpace(char c) => char.IsWhiteSpace(c) || (c >= '\x1c' && c <= '\x1f');

    static void TokenizeChunk(List<(string Text, bool Spacy)> tokens, string chunk, bool withSpecials) {
        if (TrySpecials(tokens, chunk, withSpecials)) { return; }
        var (prefixes, suffixes) = (new List<string>(), new List<string>());
        var rest = SplitAffixes(chunk, prefixes, suffixes, withSpecials);
        AttachTokens(tokens, rest, prefixes, suffixes, withSpecials);
    }

    static bool TrySpecials(List<(string Text, bool Spacy)> tokens, string chunk, bool withSpecials) {
        if (!withSpecials || !specialRules.TryGetValue(chunk, out var orths)) { return false; }
        foreach (var orth in orths) { tokens.Add((orth, false)); }
        return true;
    }

    /// <summary> Peels prefixes/suffixes off the chunk until nothing changes, bailing early whenever the remainder becomes a known special case. </summary>
    static string SplitAffixes(string chunk, List<string> prefixes, List<string> suffixes, bool withSpecials) {
        int lastSize = 0;
        while (chunk.Length != 0 && chunk.Length != lastSize) {
            if (withSpecials && specialRules.ContainsKey(chunk)) { break; }
            lastSize = chunk.Length;
            int preLen = FindPrefix(chunk);
            var minusPre = chunk[preLen..];
            if (preLen != 0 && minusPre.Length != 0 && withSpecials && specialRules.ContainsKey(minusPre)) { prefixes.Add(chunk[..preLen]); chunk = minusPre; break; }
            int sufLen = FindSuffix(minusPre);
            if (sufLen != 0) {
                var minusSuf = chunk[..^sufLen];
                if (minusSuf.Length != 0 && withSpecials && specialRules.ContainsKey(minusSuf)) { suffixes.Add(chunk[^sufLen..]); chunk = minusSuf; break; }
                if (preLen != 0 && preLen + sufLen <= chunk.Length) { prefixes.Add(chunk[..preLen]); suffixes.Add(chunk[^sufLen..]); chunk = chunk[preLen..^sufLen]; }
                else if (preLen == 0) { suffixes.Add(chunk[^sufLen..]); chunk = minusSuf; }
                else { prefixes.Add(chunk[..preLen]); chunk = minusPre; }
            } else if (preLen != 0) { prefixes.Add(chunk[..preLen]); chunk = minusPre; }
        }
        return chunk;
    }

    static int FindPrefix(string s) => prefixSearch.Match(s) is { Success: true } m ? m.Length : 0;
    static int FindSuffix(string s) => suffixSearch.Match(s) is { Success: true } m ? m.Length : 0;

    /// <summary> Emits prefixes, then the remainder (whole if special or URL, else split on infixes, skipping an infix match at position zero), then suffixes reversed. </summary>
    static void AttachTokens(List<(string Text, bool Spacy)> tokens, string rest, List<string> prefixes, List<string> suffixes, bool withSpecials) {
        foreach (var prefix in prefixes) { tokens.Add((prefix, false)); }
        if (rest.Length != 0 && !TrySpecials(tokens, rest, withSpecials)) {
            var infixes = urlMatch.IsMatch(rest) ? null : infixFinditer.Matches(rest);
            if (infixes is not { Count: > 0 }) { tokens.Add((rest, false)); }
            else {
                int start = 0;
                foreach (Match match in infixes) {
                    if (match.Index == 0) { continue; }
                    if (match.Index != start) { tokens.Add((rest[start..match.Index], false)); }
                    if (match.Length != 0) { tokens.Add((match.Value, false)); }
                    start = match.Index + match.Length;
                }
                if (start < rest.Length) { tokens.Add((rest[start..], false)); }
            }
        }
        for (int i = suffixes.Count - 1; i >= 0; i--) { tokens.Add((suffixes[i], false)); }
    }

    /// <summary> spacy's retokenization pass -- finds token runs whose concatenation equals a special case, keeps the longest non-overlapping ones, and swaps their tokens in. </summary>
    static void ApplySpecialCases(List<(string Text, bool Spacy)> tokens) {
        List<(int Start, int End)> matches = null;
        for (int i = 0; i < tokens.Count; i++) {
            if (!specialMatcher.Value.TryGetValue(tokens[i].Text, out var patterns)) { continue; }
            foreach (var orths in patterns) {
                if (i + orths.Length > tokens.Count) { continue; }
                bool matched = true;
                for (int j = 1; j < orths.Length && matched; j++) { matched = tokens[i + j].Text == orths[j]; }
                if (matched) { (matches ??= []).Add((i, i + orths.Length)); }
            }
        }
        if (matches is null) { return; }
        matches.Sort((a, b) => (a.End - a.Start) != (b.End - b.Start) ? (a.End - a.Start) - (b.End - b.Start) : b.Start - a.Start);
        var (filtered, seen) = (new List<(int Start, int End)>(), new HashSet<int>());
        for (int i = matches.Count - 1; i >= 0; i--) {
            var span = matches[i];
            if (!seen.Contains(span.Start) && !seen.Contains(span.End - 1)) { filtered.Add(span); }
            for (int j = span.Start; j < span.End; j++) { seen.Add(j); }
        }
        var byStart = filtered.ToDictionary(span => span.Start);
        var result = new List<(string Text, bool Spacy)>(tokens.Count);
        for (int i = 0; i < tokens.Count;) {
            if (!byStart.TryGetValue(i, out var span)) { result.Add(tokens[i++]); continue; }
            var sb = new StringBuilder();
            for (int j = span.Start; j < span.End; j++) { sb.Append(tokens[j].Text); if (j < span.End - 1 && tokens[j].Spacy) { sb.Append(' '); } }
            if (specialRules.TryGetValue(sb.ToString(), out var orths)) {
                foreach (var orth in orths) { result.Add((orth, false)); }
                result[^1] = (result[^1].Text, tokens[span.End - 1].Spacy);
            } else { for (int j = span.Start; j < span.End; j++) { result.Add(tokens[j]); } }
            i = span.End;
        }
        tokens.Clear();
        tokens.AddRange(result);
    }
}
