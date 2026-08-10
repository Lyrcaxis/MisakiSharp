namespace MisakiSharp;

using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;

/// <summary> Native kana-to-phoneme frontend for Japanese, ported from misaki's cutlet path (the JAG2P default Kokoro's ja voices were trained on). </summary>
/// <remarks> Text is NFKC-normalized with numbers verbalized to hiragana, segmented over unidic 3.1.0 by <see cref="JapaneseTagger"/>, then each word's kana reading maps to Kokoro's IPA. </remarks>
public static partial class JapaneseG2P {
    const string SmallKatakana = "ㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿ", FullKatakana = "クシストヌハヒフヘホムラリルレロ";
    const string Sutegana = "ゃゅょぁぃぅぇぉ", Odoriji = "〃々ゝゞヽ";
    const string PlainKana = "かきくけこさしすせそたちつてとはひふへほ", DakutenKana = "がぎぐげござじずぜぞだぢづでどばびぶべぼ";

    // misaki/cutlet.py's HEPBURN table, entries in source order (84 hiragana + 4 katakana + 76 digraphs + symbols).
    static readonly Dictionary<string, string> kanaTable =
        ("ぁ=a|あ=a|ぃ=i|い=i|ぅ=ɯ|う=ɯ|ぇ=e|え=e|ぉ=o|お=o|か=ka|が=ɡa|き=kʲi|ぎ=ɡʲi|く=kɯ|ぐ=ɡɯ|け=ke|げ=ɡe|こ=ko|ご=ɡo" +
         "|さ=sa|ざ=ʣa|し=ɕi|じ=ʥi|す=sɨ|ず=zɨ|せ=se|ぜ=ʣe|そ=so|ぞ=ʣo|た=ta|だ=da|ち=ʨi|ぢ=ʥi|つ=ʦɨ|づ=zɨ|て=te|で=de|と=to|ど=do" +
         "|な=na|に=ɲi|ぬ=nɯ|ね=ne|の=no|は=ha|ば=ba|ぱ=pa|ひ=çi|び=bʲi|ぴ=pʲi|ふ=ɸɯ|ぶ=bɯ|ぷ=pɯ|へ=he|べ=be|ぺ=pe|ほ=ho|ぼ=bo|ぽ=po" +
         "|ま=ma|み=mʲi|む=mɯ|め=me|も=mo|ゃ=ja|や=ja|ゅ=jɯ|ゆ=jɯ|ょ=jo|よ=jo|ら=ɾa|り=ɾʲi|る=ɾɯ|れ=ɾe|ろ=ɾo|ゎ=βa|わ=βa|ゐ=i|ゑ=e" +
         "|を=o|ゔ=vɯ|ゕ=ka|ゖ=ke|ヷ=va|ヸ=vʲi|ヹ=ve|ヺ=vo" +
         "|いぇ=je|うぃ=βi|うぇ=βe|うぉ=βo|きぇ=kʲe|きゃ=kʲa|きゅ=kʲɨ|きょ=kʲo|ぎゃ=ɡʲa|ぎゅ=ɡʲɨ|ぎょ=ɡʲo" +
         "|くぁ=kᵝa|くぃ=kᵝi|くぇ=kᵝe|くぉ=kᵝo|ぐぁ=ɡᵝa|ぐぃ=ɡᵝi|ぐぇ=ɡᵝe|ぐぉ=ɡᵝo|しぇ=ɕe|しゃ=ɕa|しゅ=ɕɨ|しょ=ɕo" +
         "|じぇ=ʥe|じゃ=ʥa|じゅ=ʥɨ|じょ=ʥo|ちぇ=ʨe|ちゃ=ʨa|ちゅ=ʨɨ|ちょ=ʨo|ぢゃ=ʥa|ぢゅ=ʥɨ|ぢょ=ʥo|つぁ=ʦa|つぃ=ʦʲi|つぇ=ʦe|つぉ=ʦo" +
         "|てぃ=tʲi|てゅ=tʲɨ|でぃ=dʲi|でゅ=dʲɨ|とぅ=tɯ|どぅ=dɯ|にぇ=ɲe|にゃ=ɲa|にゅ=ɲɨ|にょ=ɲo|ひぇ=çe|ひゃ=ça|ひゅ=çɨ|ひょ=ço" +
         "|びゃ=bʲa|びゅ=bʲɨ|びょ=bʲo|ぴゃ=pʲa|ぴゅ=pʲɨ|ぴょ=pʲo|ふぁ=ɸa|ふぃ=ɸʲi|ふぇ=ɸe|ふぉ=ɸo|ふゅ=ɸʲɨ|ふょ=ɸʲo" +
         "|みゃ=mʲa|みゅ=mʲɨ|みょ=mʲo|りゃ=ɾʲa|りゅ=ɾʲɨ|りょ=ɾʲo|ゔぁ=va|ゔぃ=vʲi|ゔぇ=ve|ゔぉ=vo|ゔゅ=bʲɨ|ゔょ=bʲo" +
         "|。=.|、=,|？=?|！=!|「=“|」=”|『=“|』=”|：=:|；=;|（=(|）=)|《=(|》=)|【=[|】=]|・= |，=,|～=—|〜=—|—=—|«=“|»=”|゚=|゙=")
        .Split('|').ToDictionary(entry => entry[..entry.IndexOf('=')], entry => entry[(entry.IndexOf('=') + 1)..]);

    static readonly Lazy<HashSet<string>> jaWords = new(() => {
        using var reader = new StreamReader(new GZipStream(typeof(JapaneseG2P).Assembly.GetManifestResourceStream("MisakiSharp.ja_words.txt.gz"), CompressionMode.Decompress), Encoding.UTF8);
        var words = new HashSet<string>();
        while (reader.ReadLine() is string line) { words.Add(line.Trim()); }
        return words;
    });

    /// <summary> Converts the text to Kokoro's ja phoneme string, byte-identical to misaki's `JAG2P()` (e.g. "こんにちは" -> "koɴɲiʨiha"). </summary>
    public static string Phonemize(string text) {
        if (text.Length == 0) { return ""; }
        var words = JapaneseTagger.Parse(Normalize(text))
            .Select(w => (w.Surface, Hira: KataToHira(w.Reading ?? w.Surface), CharType: w.CharType == 7 || !w.IsUnk ? 6 : w.CharType)).ToList();

        // Merge same-char_type runs whose concatenated surface is a known word (longest match), like cutlet's _romaji_tokens.
        var merged = new List<(string Surface, string Hira, int CharType)>();
        for (int i = 0; i < words.Count;) {
            int runEnd = i + 1;
            while (runEnd < words.Count && words[runEnd].CharType == words[i].CharType) { runEnd++; }
            int matchEnd = runEnd;
            while (matchEnd > i && !jaWords.Value.Contains(string.Concat(words.Skip(i).Take(matchEnd - i).Select(w => w.Surface)))) { matchEnd--; }
            if (matchEnd == i) { merged.Add(words[i]); i++; }
            else { merged.Add((string.Concat(words.Skip(i).Take(matchEnd - i).Select(w => w.Surface)), string.Concat(words.Skip(i).Take(matchEnd - i).Select(w => w.Hira)), words[i].CharType)); i = matchEnd; }
        }

        var tokens = new List<(string Roma, bool Space)>();
        foreach (var word in merged) {
            var (roma, space) = (RomajiWord(word), true);
            if ("「『«".Contains(word.Surface) || "([".Contains(roma)) { if (tokens.Count > 0) { tokens[^1] = (tokens[^1].Roma, true); } space = false; }
            else if ("」』»".Contains(word.Surface) || "]).,?!:".Contains(roma)) { if (tokens.Count > 0) { tokens[^1] = (tokens[^1].Roma, false); } }
            else if (roma == " ") { space = false; }
            tokens.Add((roma, space));
        }

        var output = string.Concat(tokens.Select(t => t.Roma.Replace("っ", "") + (t.Space ? " " : "")));
        var phonemes = Whitespace.Replace(output.Trim(), " ").Replace('(', '«').Replace(')', '»');
        return SokuonSpacing.Replace(phonemes, "");
    }

    /// <summary> cutlet's _normalize_text -- wave-dash ranges, small katakana, NFKC + mojimoji's post-NFKC residue (3 quote chars), then digit runs verbalized to hiragana. </summary>
    static string Normalize(string text) {
        text = WaveDashBeforeDigit.Replace(text, "から");
        text = string.Concat(text.Select(c => SmallKatakana.IndexOf(c) is int small && small >= 0 ? FullKatakana[small] : c));
        text = text.Normalize(NormalizationForm.FormKC).Replace('‘', '`').Replace('’', '\'').Replace('”', '"');
        var sb = new StringBuilder();
        for (int i = 0; i < text.Length;) {
            if (!char.IsDigit(text[i])) { sb.Append(text[i++]); continue; }
            int end = i + 1;
            while (end < text.Length && char.IsDigit(text[end])) { end++; }
            sb.Append(' ').Append(NumberToKana(text[i..end])); i = end;
        }
        return sb.ToString();
    }

    static string RomajiWord((string Surface, string Hira, int CharType) word) {
        if (word.Surface.All(c => c < 128)) { return word.Surface; }
        if (word.CharType == 3) { return string.Concat(word.Surface.Select(c => kanaTable.GetValueOrDefault(c.ToString(), c.ToString()))); }
        if (word.CharType != 6) { return ""; }
        return string.Concat(Enumerable.Range(0, word.Hira.Length).Select(i => MapKana(word.Hira, i)));
    }

    /// <summary> cutlet's _get_single_mapping -- one kana in context, handling odoriji, digraphs, sutegana, ー/っ and ん assimilation. </summary>
    static string MapKana(string hira, int i) {
        var (kk, pk, nk) = (hira[i], i > 0 ? hira[i - 1] : '\0', i + 1 < hira.Length ? hira[i + 1] : '\0');
        if (Odoriji.Contains(kk)) {
            if (kk is 'ゝ' or 'ヽ') { return pk == '\0' ? "" : pk.ToString(); }
            if (kk is 'ゞ') { return PlainKana.IndexOf(pk) is int plain && plain >= 0 ? kanaTable[DakutenKana[plain].ToString()] : ""; }
            return "";
        }
        if (pk != '\0' && kanaTable.TryGetValue($"{pk}{kk}", out var digraph)) { return digraph; }
        if (nk != '\0' && kanaTable.ContainsKey($"{kk}{nk}")) { return ""; }
        if (nk != '\0' && Sutegana.Contains(nk)) { return kk == 'っ' ? "" : kanaTable[kk.ToString()][..^1] + kanaTable[nk.ToString()]; }
        if (Sutegana.Contains(kk)) { return ""; }
        if (kk == 'ー') { return "ː"; }
        if (kk == 'っ') { return "ʔ"; }
        if (kk == 'ん') {
            var next = nk == '\0' ? null : kanaTable.GetValueOrDefault(nk.ToString());
            if (next is { Length: > 0 }) {
                if ("mpb".Contains(next[0])) { return "m"; }
                if ("kɡ".Contains(next[0])) { return "ŋ"; }
                if (next[0] is 'ɲ' or 'ʨ' or 'ʥ') { return "ɲ"; }
                if ("ntdɾz".Contains(next[0])) { return "n"; }
            }
            return "ɴ";
        }
        return kanaTable.GetValueOrDefault(kk.ToString(), "");
    }

    static string KataToHira(string text) => string.Concat(text.Select(c => c is >= 'ァ' and <= 'ヶ' ? (char) (c - 96) : c));

    /// <summary> misaki's num2kana Convert (hiragana) for integer strings -- including its literal too-long message, spaces pre-stripped. </summary>
    static string NumberToKana(string num) {
        if (num.Length > 9) { return "Number length too long, choose less than 10 digits"; }
        while (num.Length > 1 && num[0] == '0') { num = num[1..]; }
        return num.Length switch { 1 => One(num[0]), 2 => Two(num), 3 => Three(num), 4 => Four(num, standalone: true), _ => Big(num) };

        static string One(char digit) => "ゼロ,いち,に,さん,よん,ご,ろく,なな,はち,きゅう".Split(',')[digit - '0'];
        static string Two(string n) => n[0] == '0' ? One(n[1]) : n == "10" ? "じゅう" : n[0] == '1' ? "じゅう" + One(n[1]) : n[1] == '0' ? One(n[0]) + "じゅう" : One(n[0]) + "じゅう" + One(n[1]);
        static string Three(string n) {
            var hundreds = n[0] switch { '1' => "ひゃく", '3' => "さんびゃく", '6' => "ろっぴゃく", '8' => "はっぴゃく", var digit => One(digit) + "ひゃく" };
            return hundreds + (n[1..] == "00" ? "" : n[1] == '0' ? One(n[2]) : Two(n[1..]));
        }
        static string Four(string n, bool standalone) {
            if (n == "0000") { return ""; }
            n = n.TrimStart('0');
            if (n.Length < 4) { return n.Length switch { 1 => One(n[0]), 2 => Two(n), _ => Three(n) }; }
            var thousands = n[0] switch { '1' when standalone => "せん", '1' => "いっせん", '3' => "さんぜん", '8' => "はっせん", var digit => One(digit) + "せん" };
            return thousands + (n[1..] == "000" ? "" : n[1] == '0' ? Two(n[2..]) : Three(n[1..]));
        }
        static string Big(string n) {
            var (head, tail) = (n[..^4], Four(n[^4..], standalone: false));
            return head.Length switch {
                1 => One(head[0]) + "まん" + tail,
                2 => Two(head) + "まん" + tail,
                3 => Three(head) + "まん" + tail,
                4 => Four(head, standalone: false) + "まん" + tail,
                _ => One(head[0]) + "おく" + Four(head[1..], standalone: false) + (head[1..] == "0000" ? "" : "まん") + tail
            };
        }
    }

    static readonly Regex WaveDashBeforeDigit = new(@"[〜～](?=\d)", RegexOptions.Compiled);
    static readonly Regex Whitespace = new(@"\s+", RegexOptions.Compiled);
    static readonly Regex SokuonSpacing = new("(?<![!\",.:;?»—…”]) (?=ʔ)|(?<=ʔ) (?![\"«“])", RegexOptions.Compiled);
}
