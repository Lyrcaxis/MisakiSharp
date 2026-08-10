namespace MisakiSharp;

using System.Text;
using System.Text.RegularExpressions;

/// <summary> Native English text-to-phoneme frontend, ported 1:1 from misaki 0.9.4 (Kokoro's official G2P), bypassing espeak for known words. </summary>
/// <remarks>
/// POS tagging stays external -- inject a <see cref="Tagger"/> producing (text, tag, whitespace) triplets (misaki uses spacy `en_core_web_sm`), where
/// concatenating text+whitespace reproduces the input. Words the lexicon can't resolve go to the optional espeak fallback, or render as <c>unk</c>.
/// Supports misaki's markdown-like links: [word](2) forces stress, [word](/fənˈimz/) forces phonemes, [12](#a#) passes number flags.
/// </remarks>
public sealed partial class EnglishG2P {
    /// <summary> Tokenizes pre-processed text into (Text, Tag, Whitespace) triplets with Penn Treebank tags, exactly like spacy would. </summary>
    public delegate IReadOnlyList<(string Text, string Tag, string Whitespace)> Tagger(string text);

    /// <summary> Carries what the *next* word starts with (vowel/consonant) and whether it's "to", so earlier words can adapt (e.g. "the apple" vs "the pear"). </summary>
    public sealed record TokenContext(bool? FutureVowel = null, bool FutureTo = false);

    /// <summary> Mutable token flowing through the pipeline, mirror of misaki's MToken (with its `_` underscore extras flattened in). </summary>
    public sealed class MToken {
        public string Text, Tag, Whitespace, Phonemes;
        public string Alias, Currency, NumFlags = "";
        public bool IsHead = true, Prespace;
        public double? Stress;
        public int? Rating;
    }

    readonly record struct Word(MToken Single, List<MToken> Group);

    const char PrimaryStress = 'ˈ', SecondaryStress = 'ˌ';
    const string Vowels = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ", Consonants = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ", Diphthongs = "AIOQWYʤʧ";
    const string SubtokenJunks = "',-._‘’/", Puncts = ";:,.!?—…\"“”", NonQuotePuncts = ";:,.!?—…";
    static readonly HashSet<string> PunctTags = [".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP"];
    static readonly Dictionary<string, string> PunctTagPhonemes = new() { ["-LRB-"] = "(", ["-RRB-"] = ")", ["``"] = "“", ["\"\""] = "”", ["''"] = "”" };
    static readonly Dictionary<string, (string Unit, string Cent)> Currencies = new() { ["$"] = ("dollar", "cent"), ["£"] = ("pound", "pence"), ["€"] = ("euro", "cent") };

    readonly Lexicon lexicon;
    readonly Tagger tagger;
    readonly Func<string, string> fallback;
    readonly string unk;
    readonly bool flapsToT;

    /// <summary> The built-in spacy-equivalent tagger: native tokenization + the distilled perceptron, ready to feed <see cref="EnglishG2P"/>. </summary>
    public static IReadOnlyList<(string Text, string Tag, string Whitespace)> DefaultTagger(string text) {
        var tokens = EnglishTokenizer.Tokenize(text);
        var tags = EnglishTagger.TagTokens([.. tokens.Select(t => t.Text)]);
        return [.. tokens.Select((t, i) => (t.Text, tags[i], t.Whitespace))];
    }

    /// <param name="tagger"> Tokenizes and PTB-tags the text. Defaults to the built-in <see cref="DefaultTagger"/>. </param>
    /// <param name="espeakFallback"> Answers out-of-lexicon words. Defaults to the espeak-free <see cref="EspeakReplay"/>-backed
    /// <see cref="EspeakFallback"/> for the dialect, loaded on the first word that needs it; pass <c>_ => null</c> to get unk instead. </param>
    /// <param name="flapsToT"> Misaki rewrites 'ɾ'->'T' and 'ʔ'->'t' for every model version except 2.0 -- disable for v2 vocabs. </param>
    public EnglishG2P(Tagger tagger = null, bool british = false, Func<string, string> espeakFallback = null, string unk = "❓", bool flapsToT = true) {
        var voice = british ? "en-gb" : "en-us";
        fallback = espeakFallback ?? new EspeakFallback(british, chunk => EspeakReplay.Provider(voice)(chunk), flapsToT).Phonemize;
        (this.tagger, this.unk, this.flapsToT) = (tagger ?? DefaultTagger, unk, flapsToT);
        lexicon = new Lexicon(british);
    }

    /// <summary> Teaches this G2P custom pronunciations (e.g. brand names), registered as gold entries so they win over silver and the espeak fallback. </summary>
    /// <remarks> Capitalization variants are grown automatically ("kokoro" also covers "Kokoro"). Phonemes use misaki's IPA, e.g. "kOkˈOɹO". </remarks>
    public void AddWords(List<(string word, string phonemes)> dict) => lexicon.AddWords(dict);

    /// <summary> Converts text to misaki-style IPA, also returning the per-word tokens with their phonemes and 1-5 confidence ratings. </summary>
    public (string Phonemes, List<MToken> Tokens) Phonemize(string text) {
        var (processed, spans, features) = Preprocess(text);
        var words = Retokenize(FoldLeft(Tokenize(processed, spans, features)));
        var ctx = new TokenContext();
        for (int i = words.Count - 1; i >= 0; i--) {
            if (words[i].Single is { } single) {
                if (single.Phonemes is null) { (single.Phonemes, single.Rating) = lexicon.Phonemize(single, ctx); }
                if (single.Phonemes is null && fallback is not null) { (single.Phonemes, single.Rating) = (fallback(single.Text), 2); }
                ctx = UpdateContext(ctx, single.Phonemes, single);
                continue;
            }
            var group = words[i].Group;
            var (left, right, shouldFallback) = (0, group.Count, false);
            while (left < right) {
                var slice = group.GetRange(left, right - left);
                var merged = slice.Any(tk => tk.Alias is not null || tk.Phonemes is not null) ? null : MergeTokens(slice);
                var (ps, rating) = merged is null ? ((string)null, (int?)null) : lexicon.Phonemize(merged, ctx);
                if (ps is not null) {
                    (group[left].Phonemes, group[left].Rating) = (ps, rating);
                    for (int x = left + 1; x < right; x++) { (group[x].Phonemes, group[x].Rating) = ("", rating); }
                    ctx = UpdateContext(ctx, ps, merged);
                    (left, right) = (0, left);
                } else if (left + 1 < right) { left++; }
                else {
                    var tk = group[--right];
                    if (tk.Phonemes is null) {
                        if (tk.Text.All(SubtokenJunks.Contains)) { (tk.Phonemes, tk.Rating) = ("", 3); }
                        else if (fallback is not null) { shouldFallback = true; break; }
                    }
                    left = 0;
                }
            }
            if (shouldFallback) {
                (group[0].Phonemes, group[0].Rating) = (fallback(MergeTokens(group).Text), 2);
                for (int j = 1; j < group.Count; j++) { (group[j].Phonemes, group[j].Rating) = ("", 2); }
            } else { ResolveTokens(group); }
        }
        var tokens = words.Select(w => w.Single ?? MergeTokens(w.Group, unk)).ToList();
        if (flapsToT) { foreach (var tk in tokens) { tk.Phonemes = tk.Phonemes?.Replace('ɾ', 'T').Replace('ʔ', 't'); } }
        return (string.Concat(tokens.Select(tk => (tk.Phonemes ?? unk) + tk.Whitespace)), tokens);
    }

    /// <summary> Pulls misaki's [text](feature) links out of the text, remembering each token's span so features can be re-attached after tagging. </summary>
    static (string Text, List<(int Start, int Length)> Spans, Dictionary<int, object> Features) Preprocess(string text) {
        var (result, spans, features) = (new StringBuilder(), new List<(int, int)>(), new Dictionary<int, object>());
        text = text.TrimStart();
        int lastEnd = 0;
        foreach (Match m in LinkRegex.Matches(text)) {
            AddChunkSpans(text[lastEnd..m.Index]);
            if (ParseFeature(m.Groups[2].Value) is { } feature) { features[spans.Count] = feature; }
            spans.Add((result.Length, m.Groups[1].Value.Length));
            result.Append(m.Groups[1].Value);
            lastEnd = m.Index + m.Length;
        }
        if (lastEnd < text.Length) { AddChunkSpans(text[lastEnd..]); }
        return (result.ToString(), spans, features);

        void AddChunkSpans(string chunk) {
            for (int s = 0; s < chunk.Length;) {
                if (char.IsWhiteSpace(chunk[s])) { s++; continue; }
                int e = s + 1;
                while (e < chunk.Length && !char.IsWhiteSpace(chunk[e])) { e++; }
                spans.Add((result.Length + s, e - s));
                s = e;
            }
            result.Append(chunk);
        }
    }

    /// <summary> A feature is a stress number, /phonemes/, or #numberflags# -- anything else is silently dropped, like misaki does. </summary>
    static object ParseFeature(string f) {
        if (Lexicon.IsDigits(f.Length > 0 && f[0] is '-' or '+' ? f[1..] : f)) { return (double)int.Parse(f, System.Globalization.CultureInfo.InvariantCulture); }
        if (f is "0.5" or "+0.5") { return 0.5d; }
        if (f is "-0.5") { return -0.5d; }
        if (f.Length > 1 && f[0] == '/' && f[^1] == '/') { return '/' + f[1..].TrimEnd('/'); }
        if (f.Length > 1 && f[0] == '#' && f[^1] == '#') { return '#' + f[1..].TrimEnd('#'); }
        return null;
    }

    /// <summary> Runs the injected tagger, then routes each stored feature onto the tagger tokens overlapping its source span. </summary>
    /// <remarks>
    /// A /phonemes/ feature spanning multiple tagger tokens lands on the first one and blanks the rest, so "[New York](/nujˈɔɹk/)" stays one voiced unit.
    /// The alignment flattens per-token overlaps into one array and indexes tokens by array position, faithfully replicating misaki's np.where
    /// quirk -- when a link touches its neighbor with no whitespace, the feature shifts onto a later token exactly like the original does.
    /// </remarks>
    List<MToken> Tokenize(string text, List<(int Start, int Length)> spans, Dictionary<int, object> features) {
        var tokens = tagger(text).Select(t => new MToken() { Text = t.Text, Tag = t.Tag, Whitespace = t.Whitespace }).ToList();
        if (features.Count == 0) { return tokens; }
        var (alignData, offset) = (new List<int>(), 0);
        foreach (var tk in tokens) {
            int start = offset, end = offset + tk.Text.Length;
            alignData.AddRange(Enumerable.Range(0, spans.Count).Where(k => start < spans[k].Start + spans[k].Length && spans[k].Start < end));
            offset = end + tk.Whitespace.Length;
        }
        foreach (var (k, v) in features) {
            for (int j = 0, i = 0; j < alignData.Count; j++) {
                if (alignData[j] != k) { continue; }
                if (j < tokens.Count) {
                    if (v is double stress) { tokens[j].Stress = stress; }
                    else if (v is string s && s[0] == '/') { (tokens[j].IsHead, tokens[j].Phonemes, tokens[j].Rating) = (i == 0, i == 0 ? s.TrimStart('/') : "", 5); }
                    else if (v is string n && n[0] == '#') { tokens[j].NumFlags = n.TrimStart('#'); }
                }
                i++;
            }
        }
        return tokens;
    }

    List<MToken> FoldLeft(List<MToken> tokens) {
        var result = new List<MToken>();
        foreach (var tk in tokens) {
            if (result.Count > 0 && !tk.IsHead) { result[^1] = MergeTokens([result[^1], tk], unk); }
            else { result.Add(tk); }
        }
        return result;
    }

    /// <summary> Splits each tagger token into subtokens (CamelCase, digits, punctuation, ...), marking currency/dash/punct ones, and regroups whitespace-glued runs into word groups. </summary>
    static List<Word> Retokenize(List<MToken> tokens) {
        var words = new List<Word>();
        string currency = null;
        for (int i = 0; i < tokens.Count; i++) {
            var token = tokens[i];
            var tks = token.Alias is null && token.Phonemes is null
                ? SubtokenRegex.Matches(token.Text).Select(m => new MToken() { Text = m.Value, Tag = token.Tag, Whitespace = "", NumFlags = token.NumFlags, Stress = token.Stress }).ToList()
                : new List<MToken>() { token };
            tks[^1].Whitespace = token.Whitespace;
            for (int j = 0; j < tks.Count; j++) {
                var tk = tks[j];
                if (tk.Alias is not null || tk.Phonemes is not null) { }
                else if (tk.Tag == "$" && Currencies.ContainsKey(tk.Text)) { (currency, tk.Phonemes, tk.Rating) = (tk.Text, "", 4); }
                else if (tk.Tag == ":" && tk.Text is "-" or "–") { (tk.Phonemes, tk.Rating) = ("—", 3); }
                else if (PunctTags.Contains(tk.Tag) && tk.Text.Any(c => char.ToLowerInvariant(c) is < 'a' or > 'z')) {
                    (tk.Phonemes, tk.Rating) = (PunctTagPhonemes.GetValueOrDefault(tk.Tag) ?? string.Concat(tk.Text.Where(Puncts.Contains)), 4);
                } else if (currency is not null) {
                    if (tk.Tag != "CD") { currency = null; }
                    else if (j + 1 == tks.Count && (i + 1 == tokens.Count || tokens[i + 1].Tag != "CD")) { tk.Currency = currency; }
                } else if (j > 0 && j + 1 < tks.Count && tk.Text == "2" && char.IsLetter(tks[j - 1].Text[^1]) && char.IsLetter(tks[j + 1].Text[0])) { tk.Alias = "to"; }

                if (tk.Alias is not null || tk.Phonemes is not null) { words.Add(new(tk, null)); }
                else if (words.Count > 0 && words[^1].Group is { } group && group[^1].Whitespace == "") { tk.IsHead = false; group.Add(tk); }
                else if (tk.Whitespace != "") { words.Add(new(tk, null)); }
                else { words.Add(new(null, [tk])); }
            }
        }
        return words.Select(w => w.Group is { Count: 1 } g ? new Word(g[0], null) : w).ToList();
    }

    /// <summary> Merges consecutive tokens into one, keeping the most-capitalized text's tag and joining phonemes (with `unk` filling the unknown ones). </summary>
    static MToken MergeTokens(List<MToken> tokens, string unk = null) {
        var stresses = tokens.Where(tk => tk.Stress is not null).Select(tk => tk.Stress.Value).ToHashSet();
        var currencies = tokens.Where(tk => tk.Currency is not null).Select(tk => tk.Currency).ToHashSet();
        var ratings = tokens.Select(tk => tk.Rating).ToHashSet();
        string phonemes = null;
        if (unk is not null) {
            var sb = new StringBuilder();
            foreach (var tk in tokens) {
                if (tk.Prespace && sb.Length > 0 && !char.IsWhiteSpace(sb[^1]) && !string.IsNullOrEmpty(tk.Phonemes)) { sb.Append(' '); }
                sb.Append(tk.Phonemes ?? unk);
            }
            phonemes = sb.ToString();
        }
        return new MToken() {
            Text = string.Concat(tokens.Select((tk, i) => i + 1 == tokens.Count ? tk.Text : tk.Text + tk.Whitespace)),
            Tag = tokens.MaxBy(tk => tk.Text.Sum(c => c == char.ToLowerInvariant(c) ? 1 : 2)).Tag,
            Whitespace = tokens[^1].Whitespace,
            Phonemes = phonemes,
            IsHead = tokens[0].IsHead,
            Stress = stresses.Count == 1 ? stresses.First() : null,
            Currency = currencies.Count > 0 ? currencies.Max(StringComparer.Ordinal) : null,
            NumFlags = string.Concat(tokens.SelectMany(tk => tk.NumFlags).Distinct().Order()),
            Prespace = tokens[0].Prespace,
            Rating = ratings.Contains(null) ? null : ratings.Min(),
        };
    }

    static TokenContext UpdateContext(TokenContext ctx, string ps, MToken token) {
        bool? vowel = ctx.FutureVowel;
        foreach (var c in ps ?? "") {
            if (NonQuotePuncts.Contains(c)) { vowel = null; break; }
            if (Vowels.Contains(c)) { vowel = true; break; }
            if (Consonants.Contains(c)) { vowel = false; break; }
        }
        return new TokenContext(vowel, token.Text is "to" or "To" || (token.Text == "TO" && token.Tag is "TO" or "IN"));
    }

    /// <summary> Post-pass over a word group -- pure-junk subtokens go silent, and stress gets thinned out when over half the pieces carry primary stress. </summary>
    static void ResolveTokens(List<MToken> tokens) {
        var text = string.Concat(tokens.Select((tk, i) => i + 1 == tokens.Count ? tk.Text : tk.Text + tk.Whitespace));
        var categories = text.Where(c => !SubtokenJunks.Contains(c)).Select(c => char.IsLetter(c) ? 0 : (c is >= '0' and <= '9' ? 1 : 2)).ToHashSet();
        bool prespace = text.Contains(' ') || text.Contains('/') || categories.Count > 1;
        for (int i = 0; i < tokens.Count; i++) {
            var tk = tokens[i];
            if (tk.Phonemes is null) {
                if (i + 1 == tokens.Count && tk.Text.Length == 1 && NonQuotePuncts.Contains(tk.Text[0])) { (tk.Phonemes, tk.Rating) = (tk.Text, 3); }
                else if (tk.Text.All(SubtokenJunks.Contains)) { (tk.Phonemes, tk.Rating) = ("", 3); }
            } else if (i > 0) { tk.Prespace = prespace; }
        }
        if (prespace) { return; }
        var indices = tokens.Select((tk, i) => (HasPrimary: (tk.Phonemes ?? "").Contains(PrimaryStress), Weight: StressWeight(tk.Phonemes), Index: i))
            .Where(x => !string.IsNullOrEmpty(tokens[x.Index].Phonemes)).ToList();
        if (indices.Count == 2 && tokens[indices[0].Index].Text.Length == 1) {
            var tk = tokens[indices[1].Index];
            tk.Phonemes = ApplyStress(tk.Phonemes, -0.5);
        } else if (indices.Count >= 2 && indices.Count(x => x.HasPrimary) > (indices.Count + 1) / 2) {
            foreach (var (_, _, i) in indices.OrderBy(x => x.HasPrimary).ThenBy(x => x.Weight).ThenBy(x => x.Index).Take(indices.Count / 2)) {
                tokens[i].Phonemes = ApplyStress(tokens[i].Phonemes, -0.5);
            }
        }
    }

    static int StressWeight(string ps) => ps?.Sum(c => Diphthongs.Contains(c) ? 2 : 1) ?? 0;

    /// <summary> Remaps stress marks on the phonemes -- negative demotes/strips, positive adds, halves are the softer variants. Null passes through untouched. </summary>
    static string ApplyStress(string ps, double? stress) {
        if (stress is null) { return ps; }
        if (stress < -1) { return ps.Replace($"{PrimaryStress}", "").Replace($"{SecondaryStress}", ""); }
        if (stress == -1 || ((stress == 0 || stress == -0.5) && ps.Contains(PrimaryStress))) { return ps.Replace($"{SecondaryStress}", "").Replace(PrimaryStress, SecondaryStress); }
        if ((stress == 0 || stress == 0.5 || stress == 1) && !ps.Contains(PrimaryStress) && !ps.Contains(SecondaryStress)) {
            return ps.Any(Vowels.Contains) ? Restress(SecondaryStress + ps) : ps;
        }
        if (stress >= 1 && !ps.Contains(PrimaryStress) && ps.Contains(SecondaryStress)) { return ps.Replace(SecondaryStress, PrimaryStress); }
        if (stress > 1 && !ps.Contains(PrimaryStress) && !ps.Contains(SecondaryStress)) { return ps.Any(Vowels.Contains) ? Restress(PrimaryStress + ps) : ps; }
        return ps;
    }

    /// <summary> Floats each stress mark rightward to sit directly before the next vowel, via sort on fractional positions. </summary>
    static string Restress(string ps) {
        var chars = ps.Select((c, i) => (Position: (double)i, Char: c)).ToArray();
        for (int i = 0; i < chars.Length; i++) {
            if (chars[i].Char is PrimaryStress or SecondaryStress) { chars[i].Position = Array.FindIndex(chars, i, x => Vowels.Contains(x.Char)) - 0.5; }
        }
        return string.Concat(chars.OrderBy(x => x.Position).Select(x => x.Char));
    }

    static readonly Regex LinkRegex = new(@"\[([^\]]+)\]\(([^\)]*)\)", RegexOptions.Compiled);
    static readonly Regex SubtokenRegex = new(@"^['‘’]+|\p{Lu}(?=\p{Lu}\p{Ll})|(?:^-)?(?:\d?[,.]?\d)+|[-_]+|['‘’]{2,}|\p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu})|\p{L}+(?:['‘’]\p{L})*|[^-_\p{L}'‘’\d]|['‘’]+$", RegexOptions.Compiled);
}
