namespace MisakiSharp;

using System.IO.Compression;

/// <summary> Penn Treebank POS tagger distilled from spacy's en_core_web_sm into an averaged perceptron, so English G2P needs no python at all. </summary>
/// <remarks>
/// Loads en_tagger.bin.gz (format + training pipeline live in generators/generate_en_tagger.py), and inference here
/// is bit-exact with that script's python reference. Frequent unambiguous words resolve via a plain dictionary; everything else scores
/// 17 context features (lowercased digit-folded word forms, affixes, shape flags, neighbors, and the two previous predictions fed back
/// in) against integer weights, then argmaxes over the tag set with ordinal-string tie-breaks.
/// </remarks>
public static class EnglishTagger {
    static readonly Lazy<(string[] Tags, Dictionary<string, string> TagDict, Dictionary<string, uint[]> Weights)> model = new(Load);

    /// <summary> Weights pack as (tagIndex &lt;&lt; 16) | (ushort)weight per uint, sparse per feature -- unpacked on the fly while scoring. </summary>
    static (string[], Dictionary<string, string>, Dictionary<string, uint[]>) Load() {
        var raw = new MemoryStream();
        using (var gzip = new GZipStream(typeof(EnglishTagger).Assembly.GetManifestResourceStream("MisakiSharp.en_tagger.bin.gz"), CompressionMode.Decompress)) { gzip.CopyTo(raw); }
        raw.Position = 0;
        using var reader = new BinaryReader(raw);
        if (reader.ReadString() != "KTAG1") { throw new InvalidDataException("en_tagger.bin.gz has an unexpected header."); }
        var tags = new string[reader.ReadByte()];
        for (int i = 0; i < tags.Length; i++) { tags[i] = reader.ReadString(); }
        var tagdict = new Dictionary<string, string>();
        for (int n = reader.Read7BitEncodedInt(); n > 0; n--) { tagdict[reader.ReadString()] = tags[reader.ReadByte()]; }
        var weights = new Dictionary<string, uint[]>();
        for (int n = reader.Read7BitEncodedInt(); n > 0; n--) {
            var (key, pairs) = (reader.ReadString(), new uint[reader.ReadByte()]);
            for (int i = 0; i < pairs.Length; i++) {
                var (tag, zigzag) = (reader.ReadByte(), (uint)reader.Read7BitEncodedInt());
                pairs[i] = ((uint)tag << 16) | (ushort)(short)((int)(zigzag >> 1) ^ -(int)(zigzag & 1));
            }
            weights[key] = pairs;
        }
        return (tags, tagdict, weights);
    }

    /// <summary> Tags spacy-style tokens with Penn Treebank tags ("NN", "VBD", ...), exactly like the distilled en_core_web_sm would. </summary>
    public static string[] TagTokens(IReadOnlyList<string> tokens) {
        var (tags, tagdict, weights) = model.Value;
        var norms = new string[tokens.Count + 4];
        (norms[0], norms[1], norms[^2], norms[^1]) = ("<S2>", "<S>", "<E>", "<E2>");
        for (int i = 0; i < tokens.Count; i++) { norms[i + 2] = Normalize(tokens[i]); }
        var (result, scores) = (new string[tokens.Count], new int[tags.Length]);
        var (prev, prev2) = ("<S>", "<S2>");
        for (int i = 0; i < tokens.Count; i++) {
            if (!tagdict.TryGetValue(tokens[i], out var guess)) {
                Array.Clear(scores, 0, scores.Length);
                var (w, pw, nw) = (norms[i + 2], norms[i + 1], norms[i + 3]);
                Score("b"); Score("w=" + w); Score("s1=" + Suffix(w, 1)); Score("s2=" + Suffix(w, 2)); Score("s3=" + Suffix(w, 3));
                Score("p1=" + (w.Length == 0 ? w : w[..1])); Score("sh=" + Shape(tokens[i]));
                Score("t1=" + prev); Score("t2=" + prev2); Score("tt=" + prev2 + "|" + prev); Score("tw=" + prev + "|" + w);
                Score("pw=" + pw); Score("ps=" + Suffix(pw, 3)); Score("qw=" + norms[i]);
                Score("nw=" + nw); Score("ns=" + Suffix(nw, 3)); Score("ow=" + norms[i + 4]);
                int best = 0;
                for (int t = 1; t < scores.Length; t++) { if (scores[t] > scores[best] || (scores[t] == scores[best] && string.CompareOrdinal(tags[t], tags[best]) > 0)) { best = t; } }
                guess = tags[best];
            }
            (result[i], prev2, prev) = (guess, prev, guess);

            void Score(string feature) {
                if (!weights.TryGetValue(feature, out var pairs)) { return; }
                foreach (var packed in pairs) { scores[packed >> 16] += (short)packed; }
            }
        }
        return result;
    }

    /// <summary> Word-identity folding: lowercase invariant, ascii digits to '0', surrogate halves to U+FFFD (the python side mirrors these exact tables). </summary>
    static string Normalize(string raw) {
        Span<char> chars = raw.Length <= 64 ? stackalloc char[raw.Length] : new char[raw.Length];
        for (int i = 0; i < raw.Length; i++) {
            var c = char.IsSurrogate(raw[i]) ? '�' : char.ToLowerInvariant(raw[i]);
            chars[i] = c is >= '0' and <= '9' ? '0' : c;
        }
        return chars.ToString(); // Span<char>.ToString() returns the contents; string(Span) ctor is netstandard2.1+.
    }

    /// <summary> Case/digit/hyphen flags over the raw token -- U all-caps, F first-upper, D has-digit, H has-hyphen, '-' when none apply. </summary>
    static string Shape(string raw) {
        var (anyCased, allUpper, digit, hyphen) = (false, true, false, false);
        foreach (var c in raw) {
            if (char.IsUpper(c) || char.IsLower(c)) { (anyCased, allUpper) = (true, allUpper && char.IsUpper(c)); }
            (digit, hyphen) = (digit || c is >= '0' and <= '9', hyphen || c == '-');
        }
        var flags = (anyCased && allUpper ? "U" : "") + (raw.Length > 0 && char.IsUpper(raw[0]) ? "F" : "") + (digit ? "D" : "") + (hyphen ? "H" : "");
        return flags.Length == 0 ? "-" : flags;
    }

    static string Suffix(string s, int k) => s.Length <= k ? s : s[^k..];
}
