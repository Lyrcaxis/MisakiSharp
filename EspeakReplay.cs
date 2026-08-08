namespace MisakiSharp;

using System.Globalization;
using System.IO.Compression;
using System.Text;

/// <summary> Espeak-free phonemization for the languages misaki hands to espeak (es, fr-fr, it, pt-br, hi): replays a measured espeak-ng 1.52.0 dump instead of spawning the binary. </summary>
/// <remarks>
/// Plug straight into the espeak pipeline: <c>new EspeakG2P(EspeakReplay.Provider("es"))</c> -- no espeak binaries involved, ever.
/// A word boundary is the product of two independently measured sides: the left word supplies WHAT changes (French liaison,
/// Portuguese /s/ voicing, Spanish lenition), the right word supplies WHETHER it fires -- and that is not derivable from its
/// phonemes, since 'casa' and 'kilo' both start /k/ yet only one assimilates a preceding /n/. Deltas are plain string edits,
/// so combining diacritics, tie digraphs and liaison appends all survive. Word pairs the composition gets wrong are stored
/// verbatim, the same base-plus-exceptions shape as JapaneseTagger's connection costs.
/// Words the lexicon has never seen are answered by a letter-to-sound model rather than refused, so every chunk gets phonemes.
/// </remarks>
public sealed class EspeakReplay {
    readonly record struct Entry(string Final, string Init, string ONasal, string OLen, int Ctx, string[] Tails);
    readonly record struct Exception(string TailMedial, string HeadMedial, string TailFinal, string HeadFinal);

    const string StressMarks = "ˈˌ";
    const string Modifiers = "ːʰʲʷʼ˺";
    const char Unstressed = '.';
    const string Vowels = "aeiouyɛɔəɐɨʊɪøœɑæʌɜɤɯɵʉɶɒ";
    const string Accented = "áéíóúàèìòùâêîôûãõĩũ";
    static readonly char[] ZeroWidth = ['​', '‌', '‍', '﻿'];
    static readonly Dictionary<string, EspeakReplay> loaded = [];

    readonly byte[] blob;
    readonly int[] rows;
    readonly Dictionary<(string, string), Exception> pairs = [];
    readonly Dictionary<string, string> stressPatterns = [];
    readonly Dictionary<string, Entry> onsets = [];
    readonly Dictionary<string, string[]> codas = [];
    readonly Dictionary<string, string>[] segments = [];
    readonly (int Left, int Right)[] shapes = [];
    readonly int side;
    readonly bool hindi;
    readonly bool english;
    readonly bool british;

    /// <summary> The word section stays as the raw decompressed bytes: 2.2M parsed dictionary entries across the five
    /// languages cost ~665 MB of heap, the blob and its row offsets cost what the text weighs. Rows are emitted sorted
    /// by UTF-8 key, so lookup is a binary search over the bytes and an Entry is only materialized for words actually met. </summary>
    EspeakReplay(string language) {
        hindi = language == "hi";
        english = language is "enus" or "engb";
        british = language == "engb";
        using var gz = new GZipStream(typeof(EspeakReplay).Assembly.GetManifestResourceStream($"MisakiSharp.{language}_espeak.tsv.gz"), CompressionMode.Decompress);
        using var decompressed = new MemoryStream();
        gz.CopyTo(decompressed);
        blob = decompressed.ToArray();
        var wordRows = new List<int>();
        char section = 'W';
        for (int at = 0; at < blob.Length;) {
            int end = Array.IndexOf(blob, (byte)'\n', at);
            if (end < 0) { end = blob.Length; }
            if (end == at) { at = end + 1; continue; }
            if (blob[at] == '#') {
                var line = Encoding.UTF8.GetString(blob, at, end - at);
                if (line.StartsWith("##BIGRAMS")) { section = 'B'; }
                else if (line.StartsWith("##LTS")) {
                    section = 'L';
                    shapes = [.. line[(line.LastIndexOf(' ') + 1)..].Split(';').Select(pair => pair.Split(',')).Select(pair => (int.Parse(pair[0]), int.Parse(pair[1])))];
                    segments = [.. shapes.Select(_ => new Dictionary<string, string>())];
                    side = shapes.Max(shape => Math.Max(shape.Left, shape.Right));
                }
            }
            else if (section == 'W') { wordRows.Add(at); }
            else {
                var c = Encoding.UTF8.GetString(blob, at, end - at).Split('\t');
                switch (section) {
                    case 'B': pairs[(c[0], c[1])] = new(c[2], c[3], c[4], c[5]); break;
                    case 'L' when c[0] == "S": segments[int.Parse(c[1])][c[2]] = c[3]; break;
                    case 'L' when c[0] == "T": stressPatterns[$"{c[1]}\t{c[2]}\t{c[3]}"] = c[4]; break;
                    case 'L' when c[0] == "O": onsets[c[1]] = new("", "", c[2], c[3], int.Parse(c[4]), []); break;
                    case 'L': codas[c[1]] = c[2..]; break;
                }
            }
            at = end + 1;
        }
        rows = [.. wordRows];
    }

    /// <summary> The chunk provider: one punctuation-free chunk to the raw tie'd IPA espeak would have produced. </summary>
    public static Func<string, string> Provider(string language) => Get(language).Phonemize;

    static EspeakReplay Get(string language) {
        var code = language switch { "fr-fr" => "fr", "pt-br" => "pt", "en-us" => "enus", "en-gb" => "engb", _ => language };
        lock (loaded) { return loaded.TryGetValue(code, out var replay) ? replay : loaded[code] = new(code); }
    }

    string Phonemize(string chunk) {
        var tokens = Tokens(chunk, out var glued);
        if (tokens.Count == 0) { return ""; }
        var entries = tokens.ConvertAll(Resolve);
        var forms = new string[entries.Count];
        for (int i = 0; i < entries.Count; i++) { forms[i] = i == entries.Count - 1 ? entries[i].Final : entries[i].Init; }
        for (int i = 1; i < forms.Length; i++) {
            if (forms[i].Length == 0 || forms[i - 1].Length == 0) { continue; }
            bool last = i == forms.Length - 1;
            if (pairs.TryGetValue((tokens[i - 1], tokens[i]), out var known)) {
                forms[i - 1] = ApplyTail(forms[i - 1], last ? known.TailFinal : known.TailMedial);
                forms[i] = ApplyHead(forms[i], last ? known.HeadFinal : known.HeadMedial);
                continue;
            }
            forms[i] = ApplyHead(forms[i], IsNasal(LastPhoneme(forms[i - 1])) ? entries[i].ONasal : entries[i].OLen);
            var tails = entries[i - 1].Tails;
            if (entries[i].Ctx >= 0 && entries[i].Ctx < tails.Length) { forms[i - 1] = ApplyTail(forms[i - 1], tails[entries[i].Ctx]); }
        }
        foreach (var j in glued.Order()) {
            int left = j - 2;
            if (CompoundForm(tokens[left]) is string reduced) { forms[left] = reduced; }
            var head = forms[j].TrimStart('ˈ', 'ˌ');
            if (head.Length > 0 && Vowels.Contains(head[0]) && forms[left].Length > 0) { forms[left] = Linked(forms[left], tokens[left]); }
        }
        var built = new StringBuilder();
        for (int i = 0; i < forms.Length; i++) {
            var form = string.Join(' ', forms[i].Split(' ', StringSplitOptions.RemoveEmptyEntries));
            if (form.Length == 0) { continue; }
            if (built.Length > 0 && !glued.Contains(i)) { built.Append(' '); }
            built.Append(form);
        }
        return built.ToString();

        // Measured espeak connected-speech behavior inside hyphen compounds.
        string CompoundForm(string token) => token.ToLowerInvariant() switch {
            "in" => "ɪn", "by" => "ba^ɪ", "to" => "tə", "a" => "ɐ", "up" => "ˌʌp",
            "on" => british ? "ˌɒn" : "ˌɔn", "under" => british ? "ˌʌndə" : "ˌʌndɚ", _ => null,
        };

        string Linked(string form, string token) {
            if (form.EndsWith("^ɹ")) { return form[..^2] + "ɹ"; }
            if (!british && form[^1] == 'ɚ') { return form + "ɹ"; }
            if (british && char.ToLowerInvariant(token[^1]) == 'r') { return form + "ɹ"; }
            if (!british && form[^1] == 't') { return form[..^1] + "ɾ"; }
            return form;
        }
    }

    Entry Resolve(string token) {
        if (Found(token) is var row and >= 0) { return Parse(row); }
        if (IsPossessive(token)) {
            var final = Resolve(token[..^2]).Final;
            return new(final + PossessiveSuffix(final), final + PossessiveSuffix(final), "", "", -1, []);
        }
        return Guess(IsAllCaps(token) && !english ? token : token.ToLowerInvariant());
    }

    bool IsPossessive(string token) => english && token.Length > 2 && token[^1] is 's' && token[^2] is '\'' or '’';

    /// <summary> espeak's regular possessive morphology, extracted from 17k measured word/word's pairs per dialect --
    /// consulted only for names the lexicon lacks, since context-sensitive words carry their own measured row. </summary>
    string PossessiveSuffix(string final) => LastPhoneme(final) switch {
        'ʃ' => "ɪz",
        's' or 'z' or 'ʒ' => british ? "ɪz" : "ᵻz",
        'p' or 't' or 'k' or 'f' or 'x' => "s",
        _ => "z",
    };

    /// <summary> Exact match first; a capitalised word falls back to its first-char-lowered then all-lowered entry. An ALL-CAPS one
    /// never does in the five espeak languages (espeak spells those out letter by letter), but DOES in English -- measured espeak-en
    /// treats STATUS like status and even spells isbn out lowercase, so only the dictionary class that differs (IT, US) ships rows. </summary>
    int Found(string token) {
        foreach (var form in Variants(token)) { if (Row(form) is var row and >= 0) { return row; } }
        if (IsAllCaps(token) && !english) { return -1; }
        foreach (var form in Variants(token)) {
            if (Row(char.ToLowerInvariant(form[0]) + form[1..]) is var decap and >= 0) { return decap; }
            if (Row(form.ToLowerInvariant()) is var lower and >= 0) { return lower; }
        }
        return -1;
    }

    int Row(string key) {
        var bytes = Encoding.UTF8.GetBytes(key);
        int lo = 0, hi = rows.Length - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1, cmp = Compare(bytes, rows[mid]);
            if (cmp == 0) { return mid; }
            if (cmp < 0) { hi = mid - 1; } else { lo = mid + 1; }
        }
        return -1;
    }

    int Compare(byte[] key, int at) {
        foreach (var b in key) {
            if (blob[at] == '\t' || b != blob[at]) { return b - blob[at]; }
            at++;
        }
        return blob[at] == '\t' ? 0 : -1;
    }

    Entry Parse(int row) {
        int start = rows[row], end = Array.IndexOf(blob, (byte)'\n', start);
        var c = Encoding.UTF8.GetString(blob, start, (end < 0 ? blob.Length : end) - start).Split('\t');
        return new(c[1], c[2] == "=" ? c[1] : c[2], c[3], c[4], c[5].Length == 0 ? -1 : int.Parse(c[5]), c[6..]);
    }

    /// <summary> espeak phonemizes the typographic and the straight apostrophe identically, but a corpus writes one and a fixture the other. </summary>
    static IEnumerable<string> Variants(string token) {
        yield return token;
        if (token.Contains('\'')) { yield return token.Replace('\'', '’'); }
        else if (token.Contains('’')) { yield return token.Replace('’', '\''); }
    }

    /// <summary> A danda attached to a Hindi word is silent, and zero-width characters never surface. A token
    /// espeak knows as a whole stays whole; one it does not is cut at script and category boundaries, because
    /// espeak phonemizes "8%" exactly as "8" followed by "%". English hyphen compounds ("Mid-Atlantic") are the
    /// exception: espeak concatenates the piece phonemes with no space, so those pieces are marked glued. </summary>
    List<string> Tokens(string chunk, out HashSet<int> glued) {
        var tokens = new List<string>();
        glued = [];
        foreach (var raw in chunk.Split(default(char[]), StringSplitOptions.RemoveEmptyEntries)) {
            var core = (hindi ? raw.Trim('।', '॥') : raw);
            if (core.IndexOfAny(ZeroWidth) >= 0) { core = string.Concat(core.Where(c => Array.IndexOf(ZeroWidth, c) < 0)); }
            if (core.Length == 0) { continue; }
            if (Found(core) >= 0 || IsPossessive(core)) { tokens.Add(core); continue; }
            int start = tokens.Count;
            Pieces(core, tokens);
            if (english) { for (int j = start + 2; j < tokens.Count; j++) { if (tokens[j - 1] == "-") { glued.Add(j); } } }
        }
        return tokens;
    }

    static void Pieces(string token, List<string> into) {
        int start = -1;
        char kind = '\0';
        for (int i = 0; i <= token.Length; i++) {
            char current = i == token.Length ? '\0' : IsLetterOrMark(token[i]) ? 'L' : char.IsDigit(token[i]) ? 'N' : '\0';
            if (current != kind && start >= 0) { into.Add(token[start..i]); start = -1; }
            if (current == '\0') { if (i < token.Length) { into.Add(token[i].ToString()); } }
            else if (start < 0) { start = i; }
            kind = current;
        }
    }

    static bool IsLetterOrMark(char c) => char.IsLetter(c) || char.GetUnicodeCategory(c) is UnicodeCategory.NonSpacingMark or UnicodeCategory.SpacingCombiningMark or UnicodeCategory.EnclosingMark;

    /// <summary> Python's str.isupper(): at least one cased character and no lowercase one. </summary>
    static bool IsAllCaps(string token) {
        bool cased = false;
        foreach (var c in token) {
            if (char.IsLower(c)) { return false; }
            cased |= char.IsUpper(c);
        }
        return cased;
    }

    /// <summary> A word the lexicon never saw: the letter-to-sound model supplies the phonemes, and the boundary
    /// behaviour is borrowed from lexicon words that begin and end the same way -- a word's onset decides what it
    /// does to the word before it, its coda decides what happens to its own tail. Assuming neither would silently
    /// drop Spanish lenition and Portuguese /s/ voicing on every unknown word. </summary>
    Entry Guess(string token) {
        var emission = PredictSegments(token);
        var form = ApplyStress(emission, token);
        var bare = form.Replace("ˈ", "").Replace("ˌ", "");
        var head = onsets.TryGetValue(bare[..Math.Min(2, bare.Length)], out var wide) ? wide
                 : onsets.TryGetValue(bare[..Math.Min(1, bare.Length)], out var narrow) ? narrow : new("", "", "", "", -1, []);
        var tails = codas.TryGetValue(bare[Math.Max(0, bare.Length - 2)..], out var many) && many.Length > 0 ? many
                  : codas.TryGetValue(bare[Math.Max(0, bare.Length - 1)..], out var few) ? few : [];
        return new(form, form, head.ONasal, head.OLen, head.Ctx, tails);
    }

    /// <summary> Each letter's phonemes from the letters around it. Windows are asymmetric (a 'c' needs its RIGHT
    /// neighbour, a final 's' its LEFT) and consulted widest first. </summary>
    string[] PredictSegments(string word) {
        var padded = string.Concat(new string('#', side), word, new string('#', side));
        var emission = new string[word.Length];
        for (int i = 0; i < word.Length; i++) {
            emission[i] = "";
            for (int s = 0; s < shapes.Length; s++) {
                if (segments[s].TryGetValue(padded.Substring(i + side - shapes[s].Left, shapes[s].Left + shapes[s].Right + 1), out var chunk)) {
                    emission[i] = chunk;
                    break;
                }
            }
        }
        return emission;
    }

    /// <summary> Stress is counted in syllables from the END of the word, which no fixed letter window can see,
    /// so it is placed separately from a pattern keyed on nucleus count, accented nucleus and suffix. </summary>
    string ApplyStress(string[] emission, string word) {
        var (nuclei, accent) = StressFeatures(word, emission);
        var pattern = "";
        foreach (var key in StressKeys(nuclei, accent, word)) { if (stressPatterns.TryGetValue(key, out pattern)) { break; } }
        var built = new StringBuilder();
        int index = 0;
        foreach (var chunk in emission) {
            foreach (var unit in PhonemeUnits(chunk)) {
                var bare = unit[StressLength(unit)..];
                if (!IsNucleus(bare)) { built.Append(bare); continue; }
                char mark = index < pattern.Length ? pattern[index] : Unstressed;
                if (mark != Unstressed) { built.Append(mark); }
                built.Append(bare);
                index++;
            }
        }
        return built.ToString();
    }

    (int Nuclei, int Accent) StressFeatures(string word, string[] emission) {
        int nuclei = 0, accent = -1;
        for (int i = 0; i < emission.Length; i++) {
            foreach (var unit in PhonemeUnits(emission[i])) {
                if (!IsNucleus(unit[StressLength(unit)..])) { continue; }
                nuclei++;
                if (Accented.Contains(word[i])) { accent = nuclei; }
            }
        }
        return (nuclei, accent >= 0 ? nuclei - accent : -1);
    }

    static IEnumerable<string> StressKeys(int nuclei, int accent, string word) {
        for (int suffix = 3; suffix >= 1; suffix--) { yield return $"{nuclei}\t{accent}\t{word[Math.Max(0, word.Length - suffix)..]}"; }
        yield return $"{nuclei}\t{accent}\t";
        yield return $"-1\t{accent}\t";
        yield return "-1\t-1\t";
    }

    /// <summary> One unit per segment: leading stress marks, the base character, whatever binds to it (combining
    /// tilde, length, aspiration) and an optional '^'-tied second segment. </summary>
    static List<string> PhonemeUnits(string text) {
        var units = new List<string>();
        for (int i = 0; i < text.Length;) {
            int j = i;
            while (j < text.Length && StressMarks.Contains(text[j])) { j++; }
            if (j >= text.Length) { break; }
            j++;
            while (j < text.Length && Binds(text[j])) { j++; }
            if (j < text.Length && text[j] == '^') {
                j = Math.Min(j + 2, text.Length);
                while (j < text.Length && Binds(text[j])) { j++; }
            }
            units.Add(text[i..j]);
            i = j;
        }
        return units;
    }

    static bool Binds(char c) => c is >= '̀' and <= 'ͯ' || Modifiers.Contains(c);

    static int StressLength(string unit) {
        int k = 0;
        while (k < unit.Length && StressMarks.Contains(unit[k])) { k++; }
        return k;
    }

    static bool IsNucleus(string bare) => bare.Length > 0 && Vowels.Contains(bare[0]);

    /// <summary> Deltas are "n|text": drop n characters off the end being joined, then splice text in.
    /// A delta measured against the stressed form can overrun the shorter destressed one, so the drop is clamped. </summary>
    static string ApplyHead(string form, string delta) {
        if (delta.Length == 0) { return form; }
        int bar = delta.IndexOf('|');
        return string.Concat(delta.AsSpan(bar + 1), form.AsSpan(Math.Min(int.Parse(delta[..bar]), form.Length)));
    }

    static string ApplyTail(string form, string delta) {
        if (delta.Length == 0) { return form; }
        int bar = delta.IndexOf('|');
        return string.Concat(form.AsSpan(0, Math.Max(0, form.Length - int.Parse(delta[..bar]))), delta.AsSpan(bar + 1));
    }

    /// <summary> Last real phoneme character, skipping stress marks, espeak's '-' marker, ties and combining diacritics. </summary>
    static char LastPhoneme(string form) {
        for (int i = form.Length - 1; i >= 0; i--) {
            if (form[i] is not ('ˈ' or 'ˌ' or '-' or '^' or ' ') && form[i] is < '̀' or > 'ͯ') { return form[i]; }
        }
        return '\0';
    }

    static bool IsNasal(char phoneme) => phoneme is 'n' or 'm' or 'ŋ' or 'ɲ' or 'ɳ';
}
