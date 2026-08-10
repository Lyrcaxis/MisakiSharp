namespace MisakiSharp;

using System.Text;
using System.Text.RegularExpressions;

/// <summary> Native port of misaki's ZHFrontend (Kokoro v1.1 zh pipeline): turns Chinese text into zhuyin-style phoneme strings via pinyin initials/finals. </summary>
/// <remarks>
/// Word segmentation (jieba posseg), tone sandhi, and the pinyin lookup are injected as delegates, so this class is a pure orchestrator mirroring zh_frontend.py 1:1.
/// The pinyin delegate must act like pypinyin's lazy_pinyin with Style.INITIALS / Style.FINALS_TONE3 (neutral_tone_with_five), with large_pinyin,
/// <see cref="PinyinPhraseOverrides"/> and the '地' candidate reorder ("de,di4") loaded, and must return fresh lists (we patch them up in place).
/// </remarks>
public sealed partial class ZhFrontend(
        Func<string, List<(string word, string pos)>> posseg,
        Func<List<(string word, string pos)>, List<(string word, string pos)>> preMergeForModify,
        Func<string, string, List<string>, List<string>> modifiedTone,
        Func<string, (List<string> initials, List<string> finals)> pinyinLookup,
        string unk = "❓") {

    /// <summary> One segmented word with its POS tag, trailing whitespace ('/' marks a word boundary between adjacent spoken words), and resolved phonemes (null means unknown -> unk). </summary>
    public sealed class Token { public string Text; public string Tag; public string Whitespace = ""; public string Phonemes; }

    static readonly Dictionary<string, string> zhMap = new() {
        ["b"] = "ㄅ", ["p"] = "ㄆ", ["m"] = "ㄇ", ["f"] = "ㄈ", ["d"] = "ㄉ", ["t"] = "ㄊ", ["n"] = "ㄋ", ["l"] = "ㄌ", ["g"] = "ㄍ", ["k"] = "ㄎ", ["h"] = "ㄏ",
        ["j"] = "ㄐ", ["q"] = "ㄑ", ["x"] = "ㄒ", ["zh"] = "ㄓ", ["ch"] = "ㄔ", ["sh"] = "ㄕ", ["r"] = "ㄖ", ["z"] = "ㄗ", ["c"] = "ㄘ", ["s"] = "ㄙ",
        ["a"] = "ㄚ", ["o"] = "ㄛ", ["e"] = "ㄜ", ["ie"] = "ㄝ", ["ai"] = "ㄞ", ["ei"] = "ㄟ", ["ao"] = "ㄠ", ["ou"] = "ㄡ", ["an"] = "ㄢ", ["en"] = "ㄣ",
        ["ang"] = "ㄤ", ["eng"] = "ㄥ", ["er"] = "ㄦ", ["i"] = "ㄧ", ["u"] = "ㄨ", ["v"] = "ㄩ", ["ii"] = "ㄭ", ["iii"] = "十", ["ve"] = "月", ["ia"] = "压",
        ["ian"] = "言", ["iang"] = "阳", ["iao"] = "要", ["in"] = "阴", ["ing"] = "应", ["iong"] = "用", ["iou"] = "又", ["ong"] = "中", ["ua"] = "穵",
        ["uai"] = "外", ["uan"] = "万", ["uang"] = "王", ["uei"] = "为", ["uen"] = "文", ["ueng"] = "瓮", ["uo"] = "我", ["van"] = "元", ["vn"] = "云",
    };
    static readonly HashSet<char> punc = [.. ";:,.!?—…\"()“”"];
    static readonly HashSet<string> mustErhua = ["小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿"];
    static readonly HashSet<string> notErhua = [
        "虐儿", "为儿", "护儿", "瞒儿", "救儿", "替儿", "有儿", "一儿", "我儿", "俺儿", "妻儿",
        "拐儿", "聋儿", "乞儿", "患儿", "幼儿", "孤儿", "婴儿", "婴幼儿", "连体儿", "脑瘫儿",
        "流浪儿", "体弱儿", "混血儿", "蜜雪儿", "舫儿", "祖儿", "美儿", "应采儿", "可儿", "侄儿",
        "孙儿", "侄孙儿", "女儿", "男儿", "红孩儿", "花儿", "虫儿", "马儿", "鸟儿", "猪儿", "猫儿",
        "狗儿", "少儿",
    ];

    static ZhFrontend() { foreach (var c in ";:,.!?/—…\"()“” 12345R") { zhMap[c.ToString()] = c.ToString(); } }

    /// <summary> Pypinyin phrase overrides that zh_frontend.py loads on top of large_pinyin -- the injected pinyin lookup must honor these to match misaki. </summary>
    /// <remarks> Values are pypinyin's internal tone-inside-syllable spellings, verbatim. Misaki also reorders the single char '地' to "de,di4" (neutral 'de' wins by default). </remarks>
    public static readonly Dictionary<string, string[]> PinyinPhraseOverrides = new() {
        ["开户行"] = ["ka1i", "hu4", "hang2"], ["发卡行"] = ["fa4", "ka3", "hang2"], ["放款行"] = ["fa4ng", "kua3n", "hang2"],
        ["茧行"] = ["jia3n", "hang2"], ["行号"] = ["hang2", "ha4o"], ["各地"] = ["ge4", "di4"], ["借还款"] = ["jie4", "hua2n", "kua3n"],
        ["时间为"] = ["shi2", "jia1n", "we2i"], ["为准"] = ["we2i", "zhu3n"], ["色差"] = ["se4", "cha1"], ["嗲"] = ["dia3"],
        ["呗"] = ["bei5"], ["不"] = ["bu4"], ["咗"] = ["zuo5"], ["嘞"] = ["lei5"], ["掺和"] = ["chan1", "huo5"],
    };

    /// <summary> Converts text to the zhuyin phoneme string, e.g. "你好" -> "ㄋㄧ2ㄏㄠ3". Words separate with '/', punctuation passes through, unknowns become unk. </summary>
    public (string phonemes, List<Token> tokens) Phonemize(string text, bool withErhua = true) {
        var tokens = new List<Token>();
        foreach (var (word, tag) in preMergeForModify(posseg(text))) {
            var pos = tag switch {
                "x" when word.All(c => c is >= '一' and <= '鿿') => "X",
                not "x" when IsPunc(word) => "x",
                _ => tag,
            };
            var token = new Token() { Text = word, Tag = pos };
            if (pos is "x" or "eng") {
                if (!word.All(char.IsWhiteSpace)) { token.Phonemes = pos == "x" && IsPunc(word) ? word : null; tokens.Add(token); }
                else if (tokens.Count > 0) { tokens[^1].Whitespace += word; }
                continue;
            }
            if (tokens is [.., { Tag: not ("x" or "eng"), Whitespace: "" } previous]) { previous.Whitespace = "/"; }

            var (initials, finals) = GetInitialsFinals(word);
            finals = modifiedTone(word, pos, finals);
            if (withErhua) { (initials, finals) = MergeErhua(initials, finals, word, pos); }
            var phones = new List<string>();
            foreach (var (c, v) in initials.Zip(finals)) {
                if (c != "") { phones.Add(c); }
                if (v != "" && (!IsPunc(v) || v != c)) { phones.Add(v); }
            }
            var joined = string.Join("_", phones).Replace("_eR", "_er").Replace("R", "_R");
            token.Phonemes = string.Concat(DigitStart.Replace(joined, "_").Split('_').Select(p => zhMap.GetValueOrDefault(p, unk)));
            tokens.Add(token);
        }
        return (string.Concat(tokens.Select(t => (t.Phonemes ?? unk) + t.Whitespace)), tokens);
    }

    /// <summary> Raw pinyin initials/finals for a word, with misaki's post-fixes: '嗯' forces final "n2", and bare "i" tones split into ii (z/c/s) vs iii (zh/ch/sh/r). </summary>
    (List<string> initials, List<string> finals) GetInitialsFinals(string word) {
        var (rawInitials, rawFinals) = pinyinLookup(word);
        for (int i = 0; i < word.Length; i++) { if (word[i] == '嗯') { rawFinals[i] = "n2"; } }
        var (initials, finals) = (new List<string>(), new List<string>());
        foreach (var (c, v) in rawInitials.Zip(rawFinals)) {
            initials.Add(c);
            finals.Add(v.Length > 1 && v[0] == 'i' && char.IsDigit(v[1]) ? c switch {
                "z" or "c" or "s" => v.Replace("i", "ii"),
                "zh" or "ch" or "sh" or "r" => v.Replace("i", "iii"),
                _ => v,
            } : v);
        }
        return (initials, finals);
    }

    /// <summary> Erhua handling: trailing 儿 (er2/er5) melts into the previous final as an R before its tone digit, unless the word/pos says it's a真儿 (spoken) 儿. </summary>
    static (List<string>, List<string>) MergeErhua(List<string> initials, List<string> finals, string word, string pos) {
        if (finals.Count > 0 && word[finals.Count - 1] == '儿' && finals[^1] == "er1") { finals[^1] = "er2"; }
        if (!mustErhua.Contains(word) && (notErhua.Contains(word) || pos is "a" or "j" or "nr")) { return (initials, finals); }
        if (finals.Count != word.Length) { return (initials, finals); }
        var (newInitials, newFinals) = (new List<string>(), new List<string>());
        for (int i = 0; i < finals.Count; i++) {
            bool mergesUp = i == finals.Count - 1 && word[i] == '儿' && finals[i] is "er2" or "er5" && !notErhua.Contains(word[^Math.Min(2, word.Length)..]) && newFinals.Count > 0;
            if (mergesUp) { newFinals[^1] = newFinals[^1][..^1] + "R" + newFinals[^1][^1]; }
            else { newInitials.Add(initials[i]); newFinals.Add(finals[i]); }
        }
        return (newInitials, newFinals);
    }

    static bool IsPunc(string s) => s.Length == 1 && punc.Contains(s[0]);

    static readonly Regex DigitStart = new(@"(?=\d)", RegexOptions.Compiled);
}

/// <summary> Port of misaki zh.py's ZHG2P (v1.1 glue): arabic numbers to hanzi (cn2an "an2cn"), CJK punctuation mapping, then en/zh interleave around <see cref="ZhFrontend"/>. </summary>
/// <remarks> English spans go through the injected enCallable (null -> they collapse to unk, like misaki warns). Segments join with a single space. </remarks>
public sealed partial class ZhG2P(ZhFrontend frontend, Func<string, string> enCallable = null, string unk = "❓") {
    const string numerals = "零一二三四五六七八九";
    static readonly string[] units = ["", "十", "百", "千", "万", "十", "百", "千", "亿", "十", "百", "千", "万", "十", "百", "千"];

    public string Phonemize(string text) {
        if (text.Trim() is "") { return ""; }
        text = MapPunctuation(ConvertNumbers(text));
        var segments = new List<string>();
        foreach (Match match in EnZhInterleave.Matches(text)) {
            var (en, zh) = (match.Groups[1].Value.Trim(), match.Groups[2].Value.Trim());
            if (zh != "") { segments.Add(frontend.Phonemize(zh).phonemes); }
            else { segments.Add(enCallable is null ? unk : enCallable(en)); }
        }
        return string.Join(" ", segments);
    }

    /// <summary> Full-width/CJK punctuation to the ASCII-ish set the frontend knows, quotes/brackets normalized to “”/(), verbatim from ZHG2P.map_punctuation. </summary>
    public static string MapPunctuation(string text) => text
        .Replace("、", ", ").Replace("，", ", ").Replace("。", ". ").Replace("．", ". ")
        .Replace("！", "! ").Replace("：", ": ").Replace("；", "; ").Replace("？", "? ")
        .Replace("«", " “").Replace("»", "” ").Replace("《", " “").Replace("》", "” ")
        .Replace("「", " “").Replace("」", "” ").Replace("【", " “").Replace("】", "” ")
        .Replace("（", " (").Replace("）", ") ").Trim();

    /// <summary> Faithful port of cn2an.transform(text, "an2cn"): ranges, dates, fractions, percents, celsius, then plain numbers -- in that exact pass order. </summary>
    /// <remarks> Years read digit-by-digit ("2024年" -> 二零二四年), the rest through cn2an's quirky "low" integer algorithm (ported verbatim, string replaces and all). Numbers over 16 integer digits stay arabic, like cn2an's warn-and-skip. </remarks>
    public static string ConvertNumbers(string text) {
        text = RangeBeforeMeasureWord.Replace(text, m => m.Value.Split('-') is [var start, var end] && (NumberToHanzi(start), NumberToHanzi(end)) is (string s, string e) ? $"{s}到{e}" : m.Value);
        text = DateNumber.Replace(text, m => MonthDayDigits.Replace(YearDigits.Replace(m.Value, ym => DigitsToHanzi(ym.Value)), dm => IntegerToHanzi(dm.Value)));
        text = Fraction.Replace(text, m => m.Value.Split('/') is [var num, var den] && (IntegerToHanzi(num), IntegerToHanzi(den)) is (string n, string d) ? $"{d}分之{n}" : m.Value);
        text = Percent.Replace(text, m => NumberToHanzi(m.Value[..^1]) is string number ? $"百分之{number}" : m.Value);
        text = Celsius.Replace(text, m => NumberToHanzi(m.Value[..^1]) is string number ? $"{number}摄氏度" : m.Value);
        return PlainNumber.Replace(text, m => NumberToHanzi(m.Value) ?? m.Value);
    }

    /// <summary> an2cn "low" mode: optional '-' sign and decimal tail around <see cref="IntegerToHanzi"/>. Null when cn2an would refuse (17+ integer digits). </summary>
    static string NumberToHanzi(string number) {
        var (sign, digits) = number.StartsWith('-') ? ("负", number[1..]) : ("", number);
        var parts = digits.Split('.');
        if (IntegerToHanzi(parts[0]) is not string integer) { return null; }
        return sign + integer + (parts.Length > 1 ? "点" + DigitsToHanzi(parts[1][..Math.Min(16, parts[1].Length)]) : "");
    }

    /// <summary> cn2an's An2Cn.__integer_convert, verbatim: per-digit unit spam, then a single-pass replace chain cleans the 零s up. Quirky but battle-tested. </summary>
    static string IntegerToHanzi(string digits) {
        digits = DigitsToAscii(digits).TrimStart('0') is { Length: > 0 } trimmed ? trimmed : "0";
        if (digits.Length > 16) { return null; }
        var sb = new StringBuilder();
        for (int i = 0; i < digits.Length; i++) {
            var (d, position) = (digits[i] - '0', digits.Length - i - 1);
            if (d != 0) { sb.Append(numerals[d]).Append(units[position]); continue; }
            if (position % 4 == 0) { sb.Append('零').Append(units[position]); }
            if (i > 0 && sb[^1] != '零') { sb.Append('零'); }
        }
        var result = sb.ToString().Replace("零零", "零").Replace("零万", "万").Replace("零亿", "亿").Replace("亿万", "亿").Trim('零');
        result = WanYiZeroThousand.Replace(result, "$1$2");
        if (result.StartsWith("一十")) { result = result[1..]; }
        return result is "" ? "零" : result;
    }

    static string DigitsToHanzi(string digits) => string.Concat(DigitsToAscii(digits).Select(d => numerals[d - '0']));
    static string DigitsToAscii(string digits) => string.Concat(digits.Select(d => (char)('0' + (int)char.GetNumericValue(d))));

    static readonly Regex EnZhInterleave = new(@"([A-Za-z '-]*[A-Za-z][A-Za-z '-]*)|([^A-Za-z]+)", RegexOptions.Compiled);
    static readonly Regex RangeBeforeMeasureWord = new(@"\d+(\.\d+)?-\d+(\.\d+)?(?=(斤|克|千克|公斤|吨|米|厘米|毫米|公里|升|毫升|元|角|分|个|只|条|张|块|瓶|杯|份|本|辆|台|匹|头|位|亩|小时|分钟|秒|天|半))", RegexOptions.Compiled);
    static readonly Regex DateNumber = new(@"(\d{2,4}年)?(\d{1,2}月)?(\d{1,2}日)?", RegexOptions.Compiled);
    static readonly Regex YearDigits = new(@"\d+(?=年)", RegexOptions.Compiled);
    static readonly Regex MonthDayDigits = new(@"\d+", RegexOptions.Compiled);
    static readonly Regex Fraction = new(@"\d+/\d+", RegexOptions.Compiled);
    static readonly Regex Percent = new(@"-?(\d+\.)?\d+%", RegexOptions.Compiled);
    static readonly Regex Celsius = new(@"\d+℃", RegexOptions.Compiled);
    static readonly Regex PlainNumber = new(@"-?(\d+\.)?\d+", RegexOptions.Compiled);
    static readonly Regex WanYiZeroThousand = new(@"([万亿])零([一二三四五六七八九壹贰叁肆伍陆柒捌玖][千仟])", RegexOptions.Compiled);
}
