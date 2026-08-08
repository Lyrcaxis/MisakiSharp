namespace MisakiSharp;

using System.IO.Compression;
using System.Text;

/// <summary> MeCab-identical lattice tokenizer over unidic 3.1.0, feeding <see cref="JapaneseG2P"/> -- verified bit-exact against fugashi (misaki[ja]'s tagger) on held-out corpora. </summary>
/// <remarks> Replicates mecab's exact semantics: prefix lookup order, space absorption, unknown-word invoke/group/length rules, and viterbi's first-strictly-lower tie-breaking. </remarks>
public static class JapaneseTagger {
    const int MaxGroupingSize = 24;

    class Token { public ushort Lc, Rc; public short WCost; public string Pron; }
    class Node { public int Begin, End; public Token Token; public bool IsUnk; }
    class EndEntry { public long Cost; public ushort Rc; public Node Node; public EndEntry Prev; }

    static Dictionary<string, Token[]> lexicon;
    static Dictionary<int, int[]> lexiconLengths;
    static uint[] charInfo;  // per-codepoint CharInfo bitfield: compat mask:18 | default category:8 | length:4 | group:1 | invoke:1
    static Token[][] unkTokens;
    static readonly Lazy<bool> loaded = new(Load);

    // Connection costs: an int8 low-rank base (integer-exact, so it matches the generator bit-for-bit)
    // overridden by exact values for the ~1% of pairs real text actually consults, stored CSR by left rcAttr.
    static int rank, costShift, grandMean;
    static short[] rowMean, colMean;   // per rightLc, per leftRc
    static byte[] shiftA;
    static sbyte[] factorA, factorB;
    static int[] cacheRowOffsets;
    static ushort[] cacheCols;
    static short[] cacheVals;

    static uint Info(int codepoint) => charInfo[codepoint < 0xffff ? codepoint : 0];  // mecab's ucs2 decoder yields 0 for astral chars
    static bool IsKindOf(uint info, uint other) => (info & other & 0x3ffff) != 0;
    static int DefaultType(uint info) => (int) (info >> 18) & 0xff;
    static int UnkLength(uint info) => (int) (info >> 26) & 0xf;
    static bool UnkGroup(uint info) => (info >> 30 & 1) != 0;
    static bool UnkInvoke(uint info) => info >> 31 != 0;
    static long MatrixCost(ushort leftRc, ushort rightLc) {
        for (int lo = cacheRowOffsets[leftRc], hi = cacheRowOffsets[leftRc + 1] - 1; lo <= hi;) {
            int mid = (lo + hi) >> 1;
            if (cacheCols[mid] == rightLc) { return cacheVals[mid]; }
            if (cacheCols[mid] < rightLc) { lo = mid + 1; } else { hi = mid - 1; }
        }
        var (a, b, acc) = (rightLc * rank, leftRc * rank, 0L);
        for (int k = 0; k < rank; k++) { acc += (long) factorA[a + k] * factorB[b + k] << shiftA[k]; }
        return Math.Clamp(((acc + (1L << (costShift - 1))) >> costShift) + rowMean[rightLc] + colMean[leftRc] - grandMean, short.MinValue, short.MaxValue);
    }

    /// <summary> Segments text into unidic lexemes: (surface, pron field or null when unknown, mecab char category, unknown flag). </summary>
    public static List<(string Surface, string Reading, int CharType, bool IsUnk)> Parse(string text) {
        _ = loaded.Value;
        var (cp, charIdx) = Codepoints(text);
        int n = cp.Length;
        var ends = new List<EndEntry>[n + 1];
        ends[0] = [new() { Cost = 0, Rc = 0 }];
        uint spaceInfo = charInfo[' '];

        for (int pos = 0; pos < n; pos++) {
            if (ends[pos] is not { Count: > 0 } lefts) { continue; }
            int begin2 = pos;
            while (begin2 < n && IsKindOf(spaceInfo, Info(cp[begin2]))) { begin2++; }

            // Dictionary nodes, then unknown-word nodes, prepended exactly like mecab's linked lists.
            var nodes = new List<Node>();
            bool found = false;
            if (begin2 < n && lexiconLengths.TryGetValue(cp[begin2], out var lengths)) {
                foreach (int length in lengths) {
                    if (begin2 + length > n) { break; }
                    if (!lexicon.TryGetValue(text[charIdx[begin2]..charIdx[begin2 + length]], out var entries)) { continue; }
                    found = true;
                    foreach (var token in entries) { nodes.Insert(0, new() { Begin = begin2, End = begin2 + length, Token = token }); }
                }
            }
            if (begin2 < n) {
                uint info = Info(cp[begin2]);
                if (UnkInvoke(info) || !found) {
                    var templates = unkTokens[DefaultType(info)];
                    int runEnd = begin2 + 1, extra = 0;  // mecab counts run chars beyond the first, so groups span up to MaxGroupingSize+1
                    while (runEnd < n && IsKindOf(info, Info(cp[runEnd]))) { runEnd++; extra++; }
                    int groupEnd = 0;
                    if (UnkGroup(info)) {
                        if (extra <= MaxGroupingSize) { foreach (var token in templates) { nodes.Insert(0, new() { Begin = begin2, End = runEnd, Token = token, IsUnk = true }); } }
                        groupEnd = runEnd;
                    }
                    for (int length = 1; length <= UnkLength(info); length++) {
                        if (begin2 + length > runEnd || begin2 + length == groupEnd) { break; }
                        foreach (var token in templates) { nodes.Insert(0, new() { Begin = begin2, End = begin2 + length, Token = token, IsUnk = true }); }
                    }
                }
            }
            foreach (var node in nodes) {
                var (bestCost, bestLeft) = (long.MaxValue, (EndEntry) null);
                foreach (var left in lefts) {
                    long cost = left.Cost + MatrixCost(left.Rc, node.Token.Lc) + node.Token.WCost;
                    if (cost < bestCost) { (bestCost, bestLeft) = (cost, left); }
                }
                (ends[node.End] ??= []).Insert(0, new() { Cost = bestCost, Rc = node.Token.Rc, Node = node, Prev = bestLeft });
            }
        }

        int eosPos = n;
        while (ends[eosPos] is not { Count: > 0 }) { eosPos--; }  // eos absorbs trailing spaces, like mecab's backward scan
        var (eosCost, eosLeft) = (long.MaxValue, (EndEntry) null);
        foreach (var left in ends[eosPos]) {
            long cost = left.Cost + MatrixCost(left.Rc, 0);
            if (cost < eosCost) { (eosCost, eosLeft) = (cost, left); }
        }
        var result = new List<(string, string, int, bool)>();
        for (var entry = eosLeft; entry.Node is Node node; entry = entry.Prev) {
            result.Add((text[charIdx[node.Begin]..charIdx[node.End]], node.Token.Pron, DefaultType(Info(cp[node.Begin])), node.IsUnk));
        }
        result.Reverse();
        return result;
    }

    static (int[] cp, int[] charIdx) Codepoints(string text) {
        var (cps, idx) = (new List<int>(text.Length), new List<int>(text.Length + 1));
        for (int i = 0; i < text.Length; i += char.IsHighSurrogate(text[i]) ? 2 : 1) {
            idx.Add(i);
            cps.Add(char.ConvertToUtf32(text, i));
        }
        idx.Add(text.Length);
        return ([.. cps], [.. idx]);
    }

    static bool Load() {
        lexicon = new Dictionary<string, Token[]>(700_000);
        using (var reader = OpenTable("ja_lexicon.tsv.gz")) {
            while (reader.ReadLine() is string line) {
                var parts = line.Split('\t');
                var tokens = new Token[parts.Length - 1];
                for (int i = 1; i < parts.Length; i++) {
                    var f = parts[i].Split(',', 4);
                    tokens[i - 1] = new() { Lc = ushort.Parse(f[0]), Rc = ushort.Parse(f[1]), WCost = short.Parse(f[2]), Pron = Unescape(f[3]) };
                }
                lexicon[Unescape(parts[0])] = tokens;
            }
        }
        var lengthSets = new Dictionary<int, SortedSet<int>>();
        foreach (var surface in lexicon.Keys) {
            var (first, count) = (char.ConvertToUtf32(surface, 0), 0);
            for (int i = 0; i < surface.Length; i += char.IsHighSurrogate(surface[i]) ? 2 : 1) { count++; }
            (lengthSets.TryGetValue(first, out var set) ? set : lengthSets[first] = []).Add(count);
        }
        lexiconLengths = lengthSets.ToDictionary(kv => kv.Key, kv => kv.Value.ToArray());

        var categories = new Dictionary<string, (int id, int invoke, int group, int length)>();
        var ranges = new List<(int lo, int hi, int defaultId, uint mask)>();
        var unkLines = new List<string[]>();
        using (var reader = OpenTable("ja_char_unk.tsv.gz")) {
            while (reader.ReadLine() is string line) {
                var f = line.Split('\t');
                if (f[0] == "c") { categories[f[1]] = (int.Parse(f[2]), int.Parse(f[3]), int.Parse(f[4]), int.Parse(f[5])); }
                else if (f[0] == "r") { ranges.Add((int.Parse(f[1]), int.Parse(f[2]), int.Parse(f[3]), uint.Parse(f[4]))); }
                else { unkLines.Add(f); }
            }
        }
        static uint Pack(int id, int invoke, int group, int length, uint mask) => mask | (uint) id << 18 | (uint) length << 26 | (uint) group << 30 | (uint) invoke << 31;
        charInfo = new uint[0xffff];
        var (defaultId, defaultInvoke, defaultGroup, defaultLength) = categories["DEFAULT"];
        Array.Fill(charInfo, Pack(defaultId, defaultInvoke, defaultGroup, defaultLength, 1u << defaultId));
        foreach (var (lo, hi, id, mask) in ranges) {
            var (_, invoke, group, length) = categories.Values.Single(c => c.id == id);
            for (int c = lo; c <= Math.Min(hi, 0xfffe); c++) { charInfo[c] = Pack(id, invoke, group, length, mask); }
        }
        unkTokens = new Token[categories.Count][];
        foreach (var f in unkLines) {
            unkTokens[categories[f[1]].id] = [.. f.Skip(2).Select(spec => spec.Split(',')).Select(v => new Token() { Lc = ushort.Parse(v[0]), Rc = ushort.Parse(v[1]), WCost = short.Parse(v[2]) })];
        }

        LoadConnectionCosts();
        return true;
    }

    static void LoadConnectionCosts() {
        using var stream = new GZipStream(typeof(JapaneseTagger).Assembly.GetManifestResourceStream("MisakiSharp.ja_matrix_compact.bin.gz"), CompressionMode.Decompress);
        using var reader = new BinaryReader(stream);
        var (leftSize, rightSize) = (reader.ReadUInt16(), reader.ReadUInt16());
        (rank, costShift, grandMean) = (reader.ReadUInt16(), reader.ReadUInt16(), reader.ReadInt16());
        rowMean = Read<short>(reader, rightSize);
        colMean = Read<short>(reader, leftSize);
        shiftA = reader.ReadBytes(rank);
        factorA = Read<sbyte>(reader, rightSize * rank);
        factorB = Read<sbyte>(reader, leftSize * rank);
        cacheRowOffsets = Read<int>(reader, leftSize + 1);
        cacheCols = Read<ushort>(reader, cacheRowOffsets[leftSize]);
        cacheVals = Read<short>(reader, cacheRowOffsets[leftSize]);
    }

    /// <summary> Reads 'count' little-endian values of a blittable type, filling the buffer across the short reads a GZipStream hands out. </summary>
    static T[] Read<T>(BinaryReader reader, int count) where T : unmanaged {
        var values = new T[count];
        var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan());
        for (int read = 0; read < bytes.Length;) {
            int got = reader.Read(bytes[read..]);
            if (got == 0) { throw new EndOfStreamException($"ja_matrix_compact.bin.gz truncated at {read}/{bytes.Length} bytes"); }
            read += got;
        }
        return values;
    }

    static StreamReader OpenTable(string name)
        => new(new GZipStream(typeof(JapaneseTagger).Assembly.GetManifestResourceStream($"MisakiSharp.{name}"), CompressionMode.Decompress), Encoding.UTF8);

    static string Unescape(string s) {
        if (!s.Contains('\\')) { return s; }
        var sb = new StringBuilder(s.Length);
        for (int i = 0; i < s.Length; i++) {
            sb.Append(s[i] != '\\' ? s[i] : s[++i] switch { 't' => '\t', 'n' => '\n', 'r' => '\r', var c => c });
        }
        return sb.ToString();
    }
}
