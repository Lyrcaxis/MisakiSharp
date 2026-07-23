namespace MisakiSharp;

using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;

/// <summary> Native hanzi-to-phoneme frontend for Chinese, ported from misaki (Kokoro's official G2P), bypassing espeak entirely. </summary>
/// <remarks>
/// <see cref="Phonemize"/> runs the Kokoro v1.1 pipeline (posseg segmentation, tone sandhi, zhuyin output) -- pair it with the v1.1-zh models.
/// <see cref="PhonemizeLegacy"/> keeps the v1.0-era frontend (tone-arrow IPA, no sandhi) that Kokoro v1.0's zh voices were trained on.
/// </remarks>
public static partial class ChineseG2P {
    /// <summary> Hook for phonemizing latin spans embedded in Chinese text (misaki's en_callable). Null (the default) drops them as unk. </summary>
    /// <remarks> KokoroSharp wires this to its english pipeline on startup -- standalone users can plug <see cref="EnglishG2P"/> or anything else in. </remarks>
    public static Func<string, string> EnglishPhonemizer { get; set; }

    /// <summary> Converts the text to Kokoro v1.1's zhuyin phoneme string (e.g. "你好" -> "ㄋㄧ2ㄏㄠ3"), tone sandhi and all -- misaki's ZHG2P(version='1.1'). </summary>
    public static string Phonemize(string text) => modernG2P.Value.Phonemize(text);

    static StreamReader OpenTable(string name)
        => new(new GZipStream(typeof(ChineseG2P).Assembly.GetManifestResourceStream($"MisakiSharp.{name}"), CompressionMode.Decompress), Encoding.UTF8);

    #region Kokoro v1.1 frontend (misaki 0.9.4)

    static Dictionary<string, string[]> modernHanziToPinyin = [];
    static Dictionary<string, (string initial, string final)> syllableSplits = [];
    static int modernMaxPhraseLength = 1;
    static readonly Lazy<ZhG2P> modernG2P = new(BuildModernG2P);

    /// <summary> Loads the v2 tables and wires posseg + sandhi + pinyin into the ZhFrontend orchestrator. </summary>
    static ZhG2P BuildModernG2P() {
        var dictWords = new Dictionary<string, (double logFreq, string pos)>();
        double logTotal = 0;
        using (var reader = OpenTable("zh_g2p_v2.tsv.gz")) {
            while (reader.ReadLine() is string line) {
                var (key, value) = (line[..line.IndexOf('\t')], line[(line.IndexOf('\t') + 1)..]);
                if (key == "@@") { logTotal = Math.Log(double.Parse(value)); }
                else if (key[0] == '#') { syllableSplits[key[1..]] = (value[..value.IndexOf(',')], value[(value.IndexOf(',') + 1)..]); }
                else if (key[0] == '@') { var space = value.IndexOf(' '); dictWords[key[1..]] = (Math.Log(double.Parse(value[..space])), value[(space + 1)..]); }
                else { modernHanziToPinyin[key] = value.Split(' '); modernMaxPhraseLength = Math.Max(modernMaxPhraseLength, key.Length); }
            }
        }
        using (var hmm = OpenTable("zh_posseg_hmm.tsv.gz")) { PossegTagger.Initialize(dictWords, logTotal, hmm); }
        using (var hmm = OpenTable("zh_finalseg_hmm.tsv.gz")) { PossegTagger.InitializeFinalseg(hmm); }
        (ToneSandhi.CutForSearch, ToneSandhi.FinalsForWord) = (PossegTagger.CutForSearch, word => PinyinLookup(word).finals);
        var frontend = new ZhFrontend(PossegTagger.Cut, ToneSandhi.PreMergeForModify, ToneSandhi.ModifiedTone, PinyinLookup);
        return new ZhG2P(frontend, text => EnglishPhonemizer?.Invoke(text) ?? "❓");
    }

    /// <summary> pypinyin's lazy_pinyin equivalent: greedy longest phrase match over the dict, unknown in-range chars go silent, non-han runs pass through raw. </summary>
    static (List<string> initials, List<string> finals) PinyinLookup(string word) {
        var (initials, finals) = (new List<string>(), new List<string>());
        for (int i = 0; i < word.Length;) {
            if (!IsHanzi(word[i])) {
                int end = i + 1;
                while (end < word.Length && !IsHanzi(word[end])) { end++; }
                initials.Add(word[i..end]); finals.Add(word[i..end]);
                i = end; continue;
            }
            int length = Math.Min(modernMaxPhraseLength, word.Length - i);
            while (length > 1 && !modernHanziToPinyin.ContainsKey(word.Substring(i, length))) { length--; }
            if (modernHanziToPinyin.TryGetValue(word.Substring(i, length), out var pinyins)) {
                foreach (var syllable in pinyins) { var (initial, final) = syllableSplits[syllable]; initials.Add(initial); finals.Add(final); }
            } else { initials.Add(""); finals.Add(""); }
            i += length;
        }
        return (initials, finals);
    }

    #endregion

    #region Kokoro v1.0 legacy frontend (misaki 0.7.12)

    static readonly Lazy<(Dictionary<string, string[]> hanziToPinyin, Dictionary<string, string> pinyinToIpa, Dictionary<string, double> wordLogFreq, double logTotal, int maxPhraseLength, int maxWordLength)> legacy = new(() => {
        var (hanziToPinyin, pinyinToIpa, wordLogFreq) = (new Dictionary<string, string[]>(), new Dictionary<string, string>(), new Dictionary<string, double>());
        var (logTotal, maxPhraseLength, maxWordLength) = (0.0, 1, 1);
        using var reader = OpenTable("zh_g2p.tsv.gz");
        while (reader.ReadLine() is string line) {
            var (key, value) = (line[..line.IndexOf('\t')], line[(line.IndexOf('\t') + 1)..]);
            if (key == "@@") { logTotal = Math.Log(double.Parse(value)); }
            else if (key[0] == '#') { pinyinToIpa[key[1..]] = value; }
            else if (key[0] == '@') { wordLogFreq[key[1..]] = Math.Log(double.Parse(value)); maxWordLength = Math.Max(maxWordLength, key.Length - 1); }
            else { hanziToPinyin[key] = value.Split(' '); maxPhraseLength = Math.Max(maxPhraseLength, key.Length); }
        }
        return (hanziToPinyin, pinyinToIpa, wordLogFreq, logTotal, maxPhraseLength, maxWordLength);
    });

    /// <summary> Converts the text to v1.0-style tone-marked IPA (e.g. "你好" -> "ni↓xau↓"), leaving non-hanzi parts untouched. </summary>
    /// <remarks> Word boundaries become spaces, and matching the widest known phrase within each word resolves polyphones (e.g. 行 in 银行 vs 行走). </remarks>
    public static string PhonemizeLegacy(string text) {
        text = ConvertNumbers(NormalizePunctuation(text));
        var sb = new StringBuilder();
        for (int i = 0; i < text.Length;) {
            if (!IsHanzi(text[i])) { sb.Append(text[i++]); continue; }
            int runEnd = i + 1;
            while (runEnd < text.Length && IsHanzi(text[runEnd])) { runEnd++; }
            foreach (var word in Segment(text[i..runEnd])) { sb.Append(WordToIpa(word)).Append(' '); }
            i = runEnd;
        }
        var phonemes = WordSpaceBeforePunctuation().Replace(sb.ToString(), "$1");
        return MultiSpace().Replace(phonemes, " ").Trim();
    }

    /// <summary> Splits a hanzi run into words over the max-probability path of jieba's frequency dictionary, just like jieba does. </summary>
    static List<string> Segment(string run) {
        var (wordLogFreq, logTotal, maxWordLength) = (legacy.Value.wordLogFreq, legacy.Value.logTotal, legacy.Value.maxWordLength);
        var (best, splitAt) = (new double[run.Length + 1], new int[run.Length + 1]);
        for (int i = run.Length - 1; i >= 0; i--) {
            (best[i], splitAt[i]) = (double.NegativeInfinity, i + 1);
            for (int length = 1; length <= Math.Min(maxWordLength, run.Length - i); length++) {
                bool isKnown = wordLogFreq.TryGetValue(run.Substring(i, length), out var logFreq);
                if (!isKnown && length > 1) { continue; }
                var score = (isKnown ? logFreq : 0) - logTotal + best[i + length];
                if (score > best[i]) { (best[i], splitAt[i]) = (score, i + length); }
            }
        }
        var words = new List<string>();
        for (int i = 0; i < run.Length; i = splitAt[i]) { words.Add(run[i..splitAt[i]]); }
        return words;
    }

    static string WordToIpa(string word) {
        var sb = new StringBuilder();
        for (int i = 0; i < word.Length;) {
            int length = Math.Min(legacy.Value.maxPhraseLength, word.Length - i);
            while (length > 1 && !legacy.Value.hanziToPinyin.ContainsKey(word.Substring(i, length))) { length--; }
            if (legacy.Value.hanziToPinyin.TryGetValue(word.Substring(i, length), out var pinyins))
                foreach (var pinyin in pinyins) { sb.Append(legacy.Value.pinyinToIpa[pinyin]); }
            i += length;
        }
        return sb.ToString();
    }

    /// <summary> Converts full-width CJK punctuation to the vocab-known ASCII equivalents, as misaki does. </summary>
    static string NormalizePunctuation(string text) => text
        .Replace("、", ", ").Replace("，", ", ").Replace("。", ". ").Replace("！", "! ")
        .Replace("：", ": ").Replace("；", "; ").Replace("？", "? ")
        .Replace("«", " “").Replace("»", "” ").Replace("《", " “").Replace("》", "” ")
        .Replace("（", " (").Replace("）", ") ");

    /// <summary> Converts arabic numbers to hanzi so they can be spoken in Chinese, e.g. "114" -> "一百一十四" (yet years digit-by-digit, "2024年" -> "二零二四年"). </summary>
    static string ConvertNumbers(string text) {
        text = YearNumber().Replace(text, m => $"{DigitsToHanzi(m.Groups[1].Value)}年");
        return Number().Replace(text, m => m.Groups[2].Success ? $"{IntegerToHanzi(m.Groups[1].Value)}点{DigitsToHanzi(m.Groups[2].Value[1..])}" : IntegerToHanzi(m.Groups[1].Value));
    }

    static string DigitsToHanzi(string digits) => string.Concat(digits.Select(d => "零一二三四五六七八九"[d - '0']));

    static string IntegerToHanzi(string digits) {
        if (digits.Length > 16) { return DigitsToHanzi(digits); }
        if (digits.TrimStart('0').Length == 0) { return "零"; }

        // Chinese groups digits per 4 (万/亿), with 十/百/千 within each group, and collapses zero-runs into a single 零.
        var sb = new StringBuilder();
        var groupNames = new[] { "", "万", "亿", "万亿" };
        for (int g = (digits.Length - 1) / 4; g >= 0; g--) {
            var group = digits[Math.Max(0, digits.Length - g * 4 - 4)..(digits.Length - g * 4)];
            if (group.TrimStart('0').Length == 0) { continue; }
            if (sb.Length > 0 && (group.Length < 4 || group[0] == '0')) { sb.Append('零'); }
            var unitNames = new[] { "", "十", "百", "千" };
            for (int i = 0; i < group.Length; i++) {
                if (group[i] == '0') { if (sb.Length > 0 && sb[^1] != '零' && group[i..].TrimStart('0').Length > 0) { sb.Append('零'); } continue; }
                sb.Append("零一二三四五六七八九"[group[i] - '0']).Append(unitNames[group.Length - 1 - i]);
            }
            sb.Append(groupNames[g]);
        }
        var result = sb.ToString().TrimEnd('零');
        return result.StartsWith("一十") ? result[1..] : result;
    }

    #endregion

    static bool IsHanzi(char c) => c >= '一' && c <= '鿿';

    [GeneratedRegex(@"(\d+)年")]        private static partial Regex YearNumber();
    [GeneratedRegex(@"(\d+)(\.\d+)?")]  private static partial Regex Number();
    [GeneratedRegex(@" +([,.!?;:”)])")] private static partial Regex WordSpaceBeforePunctuation();
    [GeneratedRegex(@" {2,}")]          private static partial Regex MultiSpace();
}
