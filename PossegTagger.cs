namespace MisakiSharp;

using System.Globalization;

/// <summary> Word segmentation + POS tagging for Chinese, a faithful port of jieba.posseg's `lcut` (which misaki's zh frontend relies on). </summary>
/// <remarks> Dict words segment over the max-probability DAG path carrying their dictionary POS, while runs of leftover single chars re-segment via an HMM viterbi over (BMES, POS) states. </remarks>
public static class PossegTagger {
    const double MinFloat = -3.14e100;

    static Dictionary<string, (double logFreq, string pos)> words = [];
    static HashSet<string> wordPrefixes = [];
    static double logTotal;

    static char[] stateBmes;
    static string[] statePos;
    static double[] startLogProb;
    static double[][] transLogProb;
    static int[][] transSuccessors;
    static Dictionary<char, double>[] emitLogProb;
    static Dictionary<char, int[]> charStates = [];
    static int[] allStates;

    /// <summary> Feeds the tagger jieba's dictionary (from the '@word\tfreq POS' lines: logFreq = ln(freq), logTotal = ln(sum of all dict freqs)) and its HMM tables. </summary>
    /// <remarks> The hmm table is line-based TSV: "s\tB\tns\tlogP" start probs, "t\tB\tns\tE\tns\tlogP" transitions, "e\tB\tns\t字\tlogP" emissions, "c\t字\tB:ns,S:x,..." per-char candidate states. Floats round-trip from python repr(). </remarks>
    public static void Initialize(Dictionary<string, (double logFreq, string pos)> dictWords, double dictLogTotal, TextReader hmmTable) {
        (words, logTotal, wordPrefixes) = (dictWords, dictLogTotal, []);
        foreach (var word in words.Keys) { for (int len = 1; len < word.Length; len++) { wordPrefixes.Add(word[..len]); } }

        var startLines = new List<(char bmes, string pos, double logP)>();
        var transLines = new List<(char bmes, string pos, char bmes2, string pos2, double logP)>();
        var emitLines = new List<(char bmes, string pos, char hanzi, double logP)>();
        var charLines = new List<(char hanzi, string statesSpec)>();
        while (hmmTable.ReadLine() is string line) {
            var f = line.Split('\t');
            switch (f[0]) {
                case "s": startLines.Add((f[1][0], f[2], ParseF(f[3]))); break;
                case "t": transLines.Add((f[1][0], f[2], f[3][0], f[4], ParseF(f[5]))); break;
                case "e": emitLines.Add((f[1][0], f[2], f[3][0], ParseF(f[4]))); break;
                case "c": charLines.Add((f[1][0], f[2])); break;
                default: throw new InvalidDataException($"Unknown posseg hmm line: {line}");
            }
        }

        // States sorted like python's ('B','ns') tuples compare, so "larger index" == "larger tuple" during viterbi tie-breaks.
        startLines.Sort((a, b) => a.bmes != b.bmes ? a.bmes - b.bmes : string.CompareOrdinal(a.pos, b.pos));
        (stateBmes, statePos, startLogProb) = ([.. startLines.Select(s => s.bmes)], [.. startLines.Select(s => s.pos)], [.. startLines.Select(s => s.logP)]);
        var stateIndex = startLines.Select((s, i) => (s, i)).ToDictionary(x => (x.s.bmes, x.s.pos), x => x.i);
        int stateCount = startLines.Count;
        allStates = [.. Enumerable.Range(0, stateCount)];
        transLogProb = [.. allStates.Select(_ => { var row = new double[stateCount]; Array.Fill(row, double.NegativeInfinity); return row; })];
        var successors = allStates.Select(_ => new List<int>()).ToArray();
        foreach (var (b1, p1, b2, p2, logP) in transLines) { var (from, to) = (stateIndex[(b1, p1)], stateIndex[(b2, p2)]); transLogProb[from][to] = logP; successors[from].Add(to); }
        transSuccessors = [.. successors.Select(list => list.ToArray())];
        emitLogProb = [.. allStates.Select(_ => new Dictionary<char, double>())];
        foreach (var (b, p, hanzi, logP) in emitLines) { emitLogProb[stateIndex[(b, p)]][hanzi] = logP; }
        charStates = charLines.ToDictionary(x => x.hanzi, x => x.statesSpec.Split(',').Select(spec => stateIndex[(spec[0], spec[2..])]).ToArray());
    }

    /// <summary> Splits text into (word, POS-tag) pairs exactly like `jieba.posseg.lcut` -- hanzi/latin/digit blocks get segmented, whitespace and other symbols pass through tagged 'x'. </summary>
    public static List<(string word, string pos)> Cut(string text) {
        var result = new List<(string word, string pos)>();
        for (int i = 0; i < text.Length;) {
            if (IsBlockChar(text[i])) {
                int end = i + 1;
                while (end < text.Length && IsBlockChar(text[end])) { end++; }
                CutDag(text[i..end], result); i = end;
            } else if (text[i] == '\r' && i + 1 < text.Length && text[i + 1] == '\n') {
                result.Add(("\r\n", "x")); i += 2;
            } else if (IsPySpace(text[i])) {
                result.Add((text[i].ToString(), "x")); i++;
            } else {
                int length = char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]) ? 2 : 1;
                result.Add((text[i..(i + length)], "x")); i += length;
            }
        }
        return result;
    }

    /// <summary> Max-probability DAG segmentation over the dict, buffering leftover single chars for the HMM to re-segment -- jieba's `__cut_DAG`. </summary>
    static void CutDag(string block, List<(string word, string pos)> result) {
        var dag = new List<int>[block.Length];
        for (int k = 0; k < block.Length; k++) {
            dag[k] = [];
            for (int i = k; i < block.Length; i++) {
                var frag = block[k..(i + 1)];
                if (words.ContainsKey(frag)) { dag[k].Add(i); }
                else if (!wordPrefixes.Contains(frag)) { break; }
            }
            if (dag[k].Count == 0) { dag[k].Add(k); }
        }

        // Unknown fragments count as freq 1 (log 0), and probability ties resolve toward the longer word, like python's tuple max.
        var (best, splitAt) = (new double[block.Length + 1], new int[block.Length + 1]);
        for (int idx = block.Length - 1; idx >= 0; idx--) {
            best[idx] = double.NegativeInfinity;
            foreach (var x in dag[idx]) {
                var logFreq = words.TryGetValue(block[idx..(x + 1)], out var entry) ? entry.logFreq : 0.0;
                var score = logFreq - logTotal + best[x + 1];
                if (score >= best[idx]) { (best[idx], splitAt[idx]) = (score, x + 1); }
            }
        }

        var bufStart = 0;
        for (int i = 0; i < block.Length;) {
            int wordEnd = splitAt[i];
            if (wordEnd - i == 1) { i = wordEnd; continue; }
            FlushBuffer(block[bufStart..i], result);
            result.Add((block[i..wordEnd], TagOf(block[i..wordEnd])));
            (i, bufStart) = (wordEnd, wordEnd);
        }
        FlushBuffer(block[bufStart..], result);
    }

    /// <summary> Buffered single chars either keep their dict tag (len 1 or known word) or head into the finer HMM pass. </summary>
    static void FlushBuffer(string buffer, List<(string word, string pos)> result) {
        if (buffer.Length == 0) { return; }
        if (buffer.Length == 1) { result.Add((buffer, TagOf(buffer))); }
        else if (!words.ContainsKey(buffer)) { CutDetail(buffer, result); }
        else { foreach (var c in buffer) { result.Add((c.ToString(), TagOf(c.ToString()))); } }
    }

    /// <summary> jieba's `__cut_detail`: hanzi runs go through the HMM, digit/dot runs tag 'm', ascii-alnum runs tag 'eng', leftovers clump into 'x'. </summary>
    static void CutDetail(string chunk, List<(string word, string pos)> result) {
        for (int i = 0; i < chunk.Length;) {
            int end = i + 1;
            if (IsHan(chunk[i])) {
                while (end < chunk.Length && IsHan(chunk[end])) { end++; }
                HmmCut(chunk[i..end], result);
            } else if (IsNumChar(chunk[i])) {
                while (end < chunk.Length && IsNumChar(chunk[end])) { end++; }
                result.Add((chunk[i..end], "m"));
            } else if (IsAsciiAlnum(chunk[i])) {
                while (end < chunk.Length && IsAsciiAlnum(chunk[end])) { end++; }
                result.Add((chunk[i..end], "eng"));
            } else {
                while (end < chunk.Length && !IsHan(chunk[end]) && !IsNumChar(chunk[end]) && !IsAsciiAlnum(chunk[end])) { end++; }
                result.Add((chunk[i..end], "x"));
            }
            i = end;
        }
    }

    /// <summary> Viterbi over (BMES, POS) states, then words assemble from the B..E / S markers -- jieba's `viterbi` + `__cut`. </summary>
    /// <remarks> Missing emissions cost -3.14e100 and missing transitions -inf, and ties pick the higher state index -- both quirks straight from jieba, needed for exact parity. </remarks>
    static void HmmCut(string run, List<(string word, string pos)> result) {
        int stateCount = allStates.Length;
        var memPath = new int[run.Length][];
        var (vPrev, present) = (new double[stateCount], new bool[stateCount]);
        foreach (var y in charStates.GetValueOrDefault(run[0], allStates)) { vPrev[y] = startLogProb[y] + EmitOf(y, run[0]); present[y] = true; }

        for (int t = 1; t < run.Length; t++) {
            var (expectNext, expectList, prevStates) = (new bool[stateCount], new List<int>(), new List<int>());
            for (int y0 = 0; y0 < stateCount; y0++) {
                if (!present[y0] || transSuccessors[y0].Length == 0) { continue; }
                prevStates.Add(y0);
                foreach (var y in transSuccessors[y0]) { if (!expectNext[y]) { expectNext[y] = true; expectList.Add(y); } }
            }
            var obsStates = charStates.GetValueOrDefault(run[t], allStates).Where(y => expectNext[y]).ToArray();
            if (obsStates.Length == 0) { obsStates = expectList.Count > 0 ? [.. expectList] : allStates; }

            var (vCur, curPresent, paths) = (new double[stateCount], new bool[stateCount], new int[stateCount]);
            foreach (var y in obsStates) {
                var emitP = EmitOf(y, run[t]);
                var (bestProb, bestPrev) = (double.NegativeInfinity, -1);
                foreach (var y0 in prevStates) {
                    var prob = vPrev[y0] + transLogProb[y0][y] + emitP;
                    if (prob >= bestProb) { (bestProb, bestPrev) = (prob, y0); }
                }
                (vCur[y], curPresent[y], paths[y]) = (bestProb, true, bestPrev);
            }
            (vPrev, present, memPath[t]) = (vCur, curPresent, paths);
        }

        var (bestFinal, state) = (double.NegativeInfinity, -1);
        for (int y = 0; y < stateCount; y++) { if (present[y] && vPrev[y] >= bestFinal) { (bestFinal, state) = (vPrev[y], y); } }
        var route = new int[run.Length];
        for (int i = run.Length - 1; i >= 0; i--) { route[i] = state; if (i > 0) { state = memPath[i][state]; } }

        int begin = 0, nexti = 0;
        for (int i = 0; i < run.Length; i++) {
            var bmes = stateBmes[route[i]];
            if (bmes == 'B') { begin = i; }
            else if (bmes == 'E') { result.Add((run[begin..(i + 1)], statePos[route[i]])); nexti = i + 1; }
            else if (bmes == 'S') { result.Add((run[i].ToString(), statePos[route[i]])); nexti = i + 1; }
        }
        if (nexti < run.Length) { result.Add((run[nexti..], statePos[route[nexti]])); }
    }

    static readonly char[] bmesStates = ['B', 'E', 'M', 'S'];
    static readonly int[][] bmesPredecessors = [[1, 3], [0, 2], [2, 0], [3, 1]]; // B<-ES, E<-BM, M<-MB, S<-SE, in python's PrevStatus order
    static double[] finalsegStartLogProb;
    static double[][] finalsegTransLogProb;
    static Dictionary<char, double>[] finalsegEmitLogProb;

    /// <summary> Feeds the plain-cut BMES HMM (jieba.finalseg), which `CutForSearch` needs for out-of-dict runs. </summary>
    /// <remarks> Same TSV idea as the posseg table, just POS-less: "s\tB\tlogP", "t\tB\tE\tlogP", "e\tB\t字\tlogP". </remarks>
    public static void InitializeFinalseg(TextReader hmmTable) {
        int StateIndex(string s) => Array.IndexOf(bmesStates, s[0]);
        finalsegStartLogProb = new double[4];
        finalsegTransLogProb = [.. Enumerable.Range(0, 4).Select(_ => { var row = new double[4]; Array.Fill(row, MinFloat); return row; })];
        finalsegEmitLogProb = [.. Enumerable.Range(0, 4).Select(_ => new Dictionary<char, double>())];
        while (hmmTable.ReadLine() is string line) {
            var f = line.Split('\t');
            if (f[0] == "s") { finalsegStartLogProb[StateIndex(f[1])] = ParseF(f[2]); }
            else if (f[0] == "t") { finalsegTransLogProb[StateIndex(f[1])][StateIndex(f[2])] = ParseF(f[3]); }
            else { finalsegEmitLogProb[StateIndex(f[1])][f[2][0]] = ParseF(f[3]); }
        }
    }

    /// <summary> jieba's `cut_for_search`: plain-cut words, with their in-dict 2-grams and 3-grams emitted before each longer word. </summary>
    public static List<string> CutForSearch(string text) {
        var result = new List<string>();
        foreach (var w in CutPlain(text)) {
            if (w.Length > 2) { for (int i = 0; i < w.Length - 1; i++) { if (words.ContainsKey(w[i..(i + 2)])) { result.Add(w[i..(i + 2)]); } } }
            if (w.Length > 3) { for (int i = 0; i < w.Length - 2; i++) { if (words.ContainsKey(w[i..(i + 3)])) { result.Add(w[i..(i + 3)]); } } }
            result.Add(w);
        }
        return result;
    }

    /// <summary> jieba's default `cut` (accurate mode, HMM on) -- like the posseg pass but POS-less, with leftover runs going through finalseg. </summary>
    static List<string> CutPlain(string text) {
        var result = new List<string>();
        for (int i = 0; i < text.Length;) {
            if (IsPlainBlockChar(text[i])) {
                int end = i + 1;
                while (end < text.Length && IsPlainBlockChar(text[end])) { end++; }
                CutDagPlain(text[i..end], result); i = end;
            } else if (text[i] == '\r' && i + 1 < text.Length && text[i + 1] == '\n') {
                result.Add("\r\n"); i += 2;
            } else if (IsPySpace(text[i])) {
                result.Add(text[i].ToString()); i++;
            } else {
                result.Add(text[i].ToString()); i++;
            }
        }
        return result;
    }

    static void CutDagPlain(string block, List<string> result) {
        var dag = new List<int>[block.Length];
        for (int k = 0; k < block.Length; k++) {
            dag[k] = [];
            for (int i = k; i < block.Length; i++) {
                var frag = block[k..(i + 1)];
                if (words.ContainsKey(frag)) { dag[k].Add(i); }
                else if (!wordPrefixes.Contains(frag)) { break; }
            }
            if (dag[k].Count == 0) { dag[k].Add(k); }
        }
        var (best, splitAt) = (new double[block.Length + 1], new int[block.Length + 1]);
        for (int idx = block.Length - 1; idx >= 0; idx--) {
            best[idx] = double.NegativeInfinity;
            foreach (var x in dag[idx]) {
                var logFreq = words.TryGetValue(block[idx..(x + 1)], out var entry) ? entry.logFreq : 0.0;
                var score = logFreq - logTotal + best[x + 1];
                if (score >= best[idx]) { (best[idx], splitAt[idx]) = (score, x + 1); }
            }
        }

        var bufStart = 0;
        for (int i = 0; i < block.Length;) {
            int wordEnd = splitAt[i];
            if (wordEnd - i == 1) { i = wordEnd; continue; }
            FlushBufferPlain(block[bufStart..i], result);
            result.Add(block[i..wordEnd]);
            (i, bufStart) = (wordEnd, wordEnd);
        }
        FlushBufferPlain(block[bufStart..], result);
    }

    static void FlushBufferPlain(string buffer, List<string> result) {
        if (buffer.Length == 0) { return; }
        if (buffer.Length == 1) { result.Add(buffer); }
        else if (!words.ContainsKey(buffer)) { FinalsegCut(buffer, result); }
        else { foreach (var c in buffer) { result.Add(c.ToString()); } }
    }

    /// <summary> jieba.finalseg.cut: hanzi runs re-segment via the BMES viterbi, everything else splits off as alnum-ish chunks. </summary>
    static void FinalsegCut(string chunk, List<string> result) {
        for (int i = 0; i < chunk.Length;) {
            if (IsHan(chunk[i])) {
                int end = i + 1;
                while (end < chunk.Length && IsHan(chunk[end])) { end++; }
                FinalsegHmm(chunk[i..end], result); i = end;
            } else if (IsAsciiAlnum(chunk[i])) {
                // Mirrors finalseg's re_skip split: alnum runs (with optional .digits and %) come out whole...
                int end = i + 1;
                while (end < chunk.Length && IsAsciiAlnum(chunk[end])) { end++; }
                if (end < chunk.Length && chunk[end] == '.') { int d = end + 1; while (d < chunk.Length && char.IsAsciiDigit(chunk[d])) { d++; } if (d > end + 1) { end = d; } }
                if (end < chunk.Length && chunk[end] == '%') { end++; }
                result.Add(chunk[i..end]); i = end;
            } else {
                // ...and everything between two such runs comes out as ONE chunk, not per-char.
                int end = i + 1;
                while (end < chunk.Length && !IsHan(chunk[end]) && !IsAsciiAlnum(chunk[end])) { end++; }
                result.Add(chunk[i..end]); i = end;
            }
        }
    }

    static void FinalsegHmm(string run, List<string> result) {
        var v = new double[run.Length][];
        var path = new int[run.Length][];
        v[0] = [.. Enumerable.Range(0, 4).Select(y => finalsegStartLogProb[y] + finalsegEmitLogProb[y].GetValueOrDefault(run[0], MinFloat))];
        for (int t = 1; t < run.Length; t++) {
            (v[t], path[t]) = (new double[4], new int[4]);
            for (int y = 0; y < 4; y++) {
                var emitP = finalsegEmitLogProb[y].GetValueOrDefault(run[t], MinFloat);
                var (bestProb, bestPrev) = (double.NegativeInfinity, -1);
                foreach (var y0 in bmesPredecessors[y]) {
                    var prob = v[t - 1][y0] + finalsegTransLogProb[y0][y] + emitP;
                    if (prob >= bestProb) { (bestProb, bestPrev) = (prob, y0); }
                }
                (v[t][y], path[t][y]) = (bestProb, bestPrev);
            }
        }

        // Final state must be E or S; python's tuple-max means S wins probability ties.
        int state = v[run.Length - 1][3] >= v[run.Length - 1][1] ? 3 : 1;
        var route = new int[run.Length];
        for (int i = run.Length - 1; i >= 0; i--) { route[i] = state; if (i > 0) { state = path[i][state]; } }

        int begin = 0, nexti = 0;
        for (int i = 0; i < run.Length; i++) {
            var bmes = bmesStates[route[i]];
            if (bmes == 'B') { begin = i; }
            else if (bmes == 'E') { result.Add(run[begin..(i + 1)]); nexti = i + 1; }
            else if (bmes == 'S') { result.Add(run[i].ToString()); nexti = i + 1; }
        }
        if (nexti < run.Length) { result.Add(run[nexti..]); }
    }

    static bool IsPlainBlockChar(char c) => IsBlockChar(c) || c is '%' or '-';

    static string TagOf(string word) => words.TryGetValue(word, out var entry) ? entry.pos : "x";
    static double EmitOf(int state, char c) => emitLogProb[state].TryGetValue(c, out var logP) ? logP : MinFloat;
    static double ParseF(string repr) => double.Parse(repr, CultureInfo.InvariantCulture);

    static bool IsHan(char c) => c >= 0x4E00 && c <= 0x9FD5;
    static bool IsNumChar(char c) => c is '.' or (>= '0' and <= '9');
    static bool IsAsciiAlnum(char c) => c is (>= 'a' and <= 'z') or (>= 'A' and <= 'Z') or (>= '0' and <= '9');
    static bool IsBlockChar(char c) => IsHan(c) || IsAsciiAlnum(c) || c is '+' or '#' or '&' or '.' or '_';

    /// <summary> Python's regex \s -- slightly wider than .NET's (0x1C-0x1F file separators count too). </summary>
    static bool IsPySpace(char c) => (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680
        || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
