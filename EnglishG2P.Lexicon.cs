namespace MisakiSharp;

using System.Globalization;
using System.IO.Compression;
using System.Numerics;
using System.Text;
using System.Text.RegularExpressions;

public sealed partial class EnglishG2P {
    /// <summary> Dictionary-based word-to-phonemes resolver -- gold entries are curated (rating 4), silver are generated (rating 3), spellouts get 3, unknowns null. </summary>
    /// <remarks> Entries are either a plain phoneme string or a per-POS-tag table with a DEFAULT. Ratings ride along every result so callers can judge confidence. </remarks>
    public sealed partial class Lexicon {
        readonly bool british;
        readonly Dictionary<string, object> golds, silvers;
        readonly (double Some, double All) capStresses = (0.5, 2);

        const string UsTaus = "AIOWYiuæɑəɛɪɹʊʌ";
        static readonly Dictionary<string, string> AddSymbols = new() { ["."] = "dot", ["/"] = "slash" };
        static readonly Dictionary<string, string> Symbols = new() { ["%"] = "percent", ["&"] = "and", ["+"] = "plus", ["@"] = "at" };
        static readonly HashSet<string> Ordinals = ["st", "nd", "rd", "th"];

        public Lexicon(bool british) {
            this.british = british;
            (golds, silvers) = Load($"en_lexicon_{(british ? "gb" : "us")}.tsv.gz");
        }

        /// <summary> Reads the embedded pre-grown lexicon: '##SECTION' markers, then `word\tps` or `word\tTAG=ps\tTAG=ps...` lines ('TAG=' empty means null). </summary>
        static (Dictionary<string, object> golds, Dictionary<string, object> silvers) Load(string name) {
            var sections = new Dictionary<string, Dictionary<string, object>>();
            Dictionary<string, object> current = null;
            using var stream = typeof(EnglishG2P).Assembly.GetManifestResourceStream($"MisakiSharp.{name}");
            using var reader = new StreamReader(new GZipStream(stream, CompressionMode.Decompress), Encoding.UTF8);
            while (reader.ReadLine() is string line) {
                if (line.StartsWith("##")) { current = sections[line[2..]] = []; continue; }
                if (line.Length == 0 || line[0] == '#') { continue; }
                var fields = line.Split('\t');
                current[fields[0]] = fields.Length == 2 && !fields[1].Contains('=') ? fields[1]
                    : fields.Skip(1).Select(c => c.Split('=', 2)).ToDictionary(t => t[0], t => t[1].Length > 0 ? t[1] : null);
            }
            return (sections["GOLD"], sections["SILVER"]);
        }

        /// <summary> Registers custom pronunciations as gold entries, so they win over silver and the espeak fallback. </summary>
        /// <remarks> Grows capitalization variants like misaki's grow_dictionary ("kokoro" also covers "Kokoro"), with explicitly listed words winning over grown ones. </remarks>
        public void AddWords(List<(string word, string phonemes)> dict) {
            foreach (var (word, ps) in dict) { if (GrowVariant(word) is string variant) { golds[variant] = ps; } }
            foreach (var (word, ps) in dict) { golds[word] = ps; }
        }

        /// <summary> The capitalization variant misaki's grow_dictionary would add ("word" == "Word"), or null when there's none. </summary>
        static string GrowVariant(string word) {
            if (word.Length < 2) { return null; }
            var (lower, capitalized) = (word.ToLower(), char.ToUpper(word[0]) + word[1..].ToLower());
            return word == lower ? (word != capitalized ? capitalized : null) : (word == capitalized ? lower : null);
        }

        /// <summary> Resolves one token to (phonemes, rating), or (null, null) when the word is out of reach -- the caller decides on fallback. </summary>
        public (string Ps, int? Rating) Phonemize(MToken tk, TokenContext ctx) {
            var word = (tk.Alias ?? tk.Text).Replace('‘', '\'').Replace('’', '\'');
            word = string.Concat(word.Normalize(NormalizationForm.FormKC).Select(NumericIfNeeded));
            double? stress = word == word.ToLowerInvariant() ? null : (word == word.ToUpperInvariant() ? capStresses.All : capStresses.Some);
            var (ps, rating) = GetWord(word, tk.Tag, stress, ctx);
            if (ps is not null) { return (ApplyStress(AppendCurrency(ps, tk.Currency), tk.Stress), rating); }
            if (IsNumber(word, tk.IsHead)) {
                (ps, rating) = GetNumber(word, tk.Currency, tk.IsHead, tk.NumFlags);
                return (ApplyStress(ps, tk.Stress), rating);
            }
            return (null, null);
        }

        (string, int?) GetWord(string word, string tag, double? stress, TokenContext ctx) {
            var (ps, rating) = GetSpecialCase(word, tag, stress, ctx);
            if (ps is not null) { return (ps, rating); }
            var wl = word.ToLowerInvariant();
            if (word.Length > 1 && IsAlpha(word.Replace("'", "")) && word != wl && (tag != "NNP" || word.Length > 7)
                && !golds.ContainsKey(word) && !silvers.ContainsKey(word) && (word == word.ToUpperInvariant() || word[1..] == word[1..].ToLowerInvariant())
                && (golds.ContainsKey(wl) || silvers.ContainsKey(wl) || StemS(wl, tag, stress, ctx).Item1 is { Length: > 0 }
                    || StemEd(wl, tag, stress, ctx).Item1 is { Length: > 0 } || StemIng(wl, tag, stress, ctx).Item1 is { Length: > 0 })) {
                word = wl;
            }
            if (IsKnown(word, tag)) { return Lookup(word, tag, stress, ctx); }
            if (word.EndsWith("s'") && IsKnown(word[..^2] + "'s", tag)) { return Lookup(word[..^2] + "'s", tag, stress, ctx); }
            if (word.EndsWith('\'') && IsKnown(word[..^1], tag)) { return Lookup(word[..^1], tag, stress, ctx); }
            var stemmed = StemS(word, tag, stress, ctx);
            if (stemmed.Item1 is null) { stemmed = StemEd(word, tag, stress, ctx); }
            if (stemmed.Item1 is null) { stemmed = StemIng(word, tag, stress ?? 0.5, ctx); }
            return stemmed;
        }

        (string, int?) GetSpecialCase(string word, string tag, double? stress, TokenContext ctx) {
            if (tag == "ADD" && AddSymbols.TryGetValue(word, out var addSymbol)) { return Lookup(addSymbol, null, -0.5, ctx); }
            if (Symbols.TryGetValue(word, out var symbol)) { return Lookup(symbol, null, null, ctx); }
            if (word.Trim('.').Contains('.') && IsAlpha(word.Replace(".", "")) && word.Split('.').Max(s => s.Length) < 3) { return GetNNP(word); }
            if (word is "a" or "A") { return (tag == "DT" ? "ɐ" : "ˈA", 4); }
            if (word is "am" or "Am" or "AM") {
                if (tag.StartsWith("NN")) { return GetNNP(word); }
                if (ctx.FutureVowel is null || word != "am" || stress > 0) { return ((string)golds["am"], 4); }
                return ("ɐm", 4);
            }
            if (word is "an" or "An" or "AN") { return word == "AN" && tag.StartsWith("NN") ? GetNNP(word) : ("ɐn", 4); }
            if (word == "I" && tag == "PRP") { return ($"{SecondaryStress}I", 4); }
            if (word is "by" or "By" or "BY" && GetParentTag(tag) == "ADV") { return ("bˈI", 4); }
            if (word is "to" or "To" || (word == "TO" && tag is "TO" or "IN")) {
                return (ctx.FutureVowel switch { null => (string)golds["to"], false => "tə", true => "tʊ" }, 4);
            }
            if (word is "in" or "In" || (word == "IN" && tag != "NNP")) {
                return ((ctx.FutureVowel is null || tag != "IN" ? $"{PrimaryStress}" : "") + "ɪn", 4);
            }
            if (word is "the" or "The" || (word == "THE" && tag == "DT")) { return (ctx.FutureVowel == true ? "ði" : "ðə", 4); }
            if (tag == "IN" && VsRegex.IsMatch(word)) { return Lookup("versus", null, null, ctx); }
            if (word is "used" or "Used" or "USED") {
                var used = (Dictionary<string, string>)golds["used"];
                return (tag is "VBD" or "JJ" && ctx.FutureTo ? used["VBD"] : used["DEFAULT"], 4);
            }
            return (null, null);
        }

        static string GetParentTag(string tag) => tag switch {
            null => null,
            _ when tag.StartsWith("VB") => "VERB",
            _ when tag.StartsWith("NN") => "NOUN",
            _ when tag.StartsWith("ADV") || tag.StartsWith("RB") => "ADV",
            _ when tag.StartsWith("ADJ") || tag.StartsWith("JJ") => "ADJ",
            _ => tag,
        };

        /// <summary> Spells the word out letter-by-letter (acronym style), softening every stress except the last, which turns primary. </summary>
        (string, int?) GetNNP(string word) {
            var letters = word.Where(char.IsLetter).Select(c => golds.GetValueOrDefault($"{char.ToUpperInvariant(c)}") as string).ToList();
            if (letters.Any(p => p is null)) { return (null, null); }
            var ps = ApplyStress(string.Concat(letters), 0);
            int last = ps.LastIndexOf(SecondaryStress);
            return (last < 0 ? ps : $"{ps[..last]}{PrimaryStress}{ps[(last + 1)..]}", 3);
        }

        bool IsKnown(string word, string tag) {
            if (golds.ContainsKey(word) || Symbols.ContainsKey(word) || silvers.ContainsKey(word)) { return true; }
            if (!IsAlpha(word) || !word.All(c => c is '\'' or '-' or (>= 'A' and <= 'Z') or (>= 'a' and <= 'z'))) { return false; }
            if (word.Length == 1) { return true; }
            if (word == word.ToUpperInvariant() && golds.ContainsKey(word.ToLowerInvariant())) { return true; }
            return word[1..] == word[1..].ToUpperInvariant();
        }

        (string, int?) Lookup(string word, string tag, double? stress, TokenContext ctx) {
            bool isNNP = false;
            if (word == word.ToUpperInvariant() && !golds.ContainsKey(word)) {
                word = word.ToLowerInvariant(); // known words ("NOT", "REALLY") read as the word
                isNNP = tag == "NNP" && !golds.ContainsKey(word) && !silvers.ContainsKey(word);
            }
            var (entry, rating) = (golds.GetValueOrDefault(word), (int?)4);
            if (entry is null && !isNNP) { (entry, rating) = (silvers.GetValueOrDefault(word), 3); }
            var ps = entry as string;
            if (entry is Dictionary<string, string> tagged) {
                if (ctx is { FutureVowel: null } && tagged.ContainsKey("None")) { tag = "None"; }
                else if (tag is null || !tagged.ContainsKey(tag)) { tag = GetParentTag(tag); }
                ps = tag is not null && tagged.TryGetValue(tag, out var byTag) ? byTag : tagged["DEFAULT"];
            }
            if (ps is null || (isNNP && !ps.Contains(PrimaryStress))) {
                var nnp = GetNNP(word);
                if (nnp.Item1 is not null) { return nnp; }
            }
            return (ApplyStress(ps, stress), rating);
        }

        string AppendS(string stem) {
            if (string.IsNullOrEmpty(stem)) { return null; }
            if ("ptkfθ".Contains(stem[^1])) { return stem + "s"; }
            if ("szʃʒʧʤ".Contains(stem[^1])) { return stem + (british ? "ɪz" : "ᵻz"); }
            return stem + "z";
        }

        string AppendEd(string stem) {
            if (string.IsNullOrEmpty(stem)) { return null; }
            if ("pkfθʃsʧ".Contains(stem[^1])) { return stem + "t"; }
            if (stem[^1] == 'd') { return stem + (british ? "ɪd" : "ᵻd"); }
            if (stem[^1] != 't') { return stem + "d"; }
            if (british || stem.Length < 2) { return stem + "ɪd"; }
            if (UsTaus.Contains(stem[^2])) { return stem[..^1] + "ɾᵻd"; }
            return stem + "ᵻd";
        }

        string AppendIng(string stem) {
            if (string.IsNullOrEmpty(stem)) { return null; }
            if (british && "əː".Contains(stem[^1])) { return null; }
            if (!british && stem.Length > 1 && stem[^1] == 't' && UsTaus.Contains(stem[^2])) { return stem[..^1] + "ɾɪŋ"; }
            return stem + "ɪŋ";
        }

        (string, int?) StemS(string word, string tag, double? stress, TokenContext ctx) {
            string stem = null;
            if (word.Length < 3 || !word.EndsWith('s')) { return (null, null); }
            if (!word.EndsWith("ss") && IsKnown(word[..^1], tag)) { stem = word[..^1]; }
            else if ((word.EndsWith("'s") || (word.Length > 4 && word.EndsWith("es") && !word.EndsWith("ies"))) && IsKnown(word[..^2], tag)) { stem = word[..^2]; }
            else if (word.Length > 4 && word.EndsWith("ies") && IsKnown(word[..^3] + "y", tag)) { stem = word[..^3] + "y"; }
            else { return (null, null); }
            var (ps, rating) = Lookup(stem, tag, stress, ctx);
            return (AppendS(ps), rating);
        }

        (string, int?) StemEd(string word, string tag, double? stress, TokenContext ctx) {
            string stem = null;
            if (word.Length < 4 || !word.EndsWith('d')) { return (null, null); }
            if (!word.EndsWith("dd") && IsKnown(word[..^1], tag)) { stem = word[..^1]; }
            else if (word.Length > 4 && word.EndsWith("ed") && !word.EndsWith("eed") && IsKnown(word[..^2], tag)) { stem = word[..^2]; }
            else { return (null, null); }
            var (ps, rating) = Lookup(stem, tag, stress, ctx);
            return (AppendEd(ps), rating);
        }

        (string, int?) StemIng(string word, string tag, double? stress, TokenContext ctx) {
            string stem = null;
            if (word.Length < 5 || !word.EndsWith("ing")) { return (null, null); }
            if (word.Length > 5 && IsKnown(word[..^3], tag)) { stem = word[..^3]; }
            else if (IsKnown(word[..^3] + "e", tag)) { stem = word[..^3] + "e"; }
            else if (word.Length > 5 && DoubledIngRegex.IsMatch(word) && IsKnown(word[..^4], tag)) { stem = word[..^4]; }
            else { return (null, null); }
            var (ps, rating) = Lookup(stem, tag, stress, ctx);
            return (AppendIng(ps), rating);
        }

        /// <summary> Misaki compares the cents' char-set against int 0 (never equal), so in practice this only rejects 3+ decimal digits. </summary>
        static bool IsCurrency(string word) {
            if (!word.Contains('.')) { return true; }
            if (word.Count(c => c == '.') > 1) { return false; }
            return word.Split('.')[1].Length < 3;
        }

        static readonly string[] NumSuffixes = ["ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"];
        static bool IsNumber(string word, bool isHead) {
            if (word.All(c => c is < '0' or > '9')) { return false; }
            foreach (var s in NumSuffixes) {
                if (word.EndsWith(s)) { word = word[..^s.Length]; break; }
            }
            return word.Select((c, i) => c is (>= '0' and <= '9') or ',' or '.' || (isHead && i == 0 && c == '-')).All(b => b);
        }

        /// <summary> Reads out a numeric token -- cardinals, ordinals, 4-digit years, decimals, currency amounts, digit-by-digit runs, plus s/ed/ing suffixes. </summary>
        /// <remarks> Number flags tune the wording, like misaki's: 'a' says "a hundred" over "one hundred", '&'/'n' keep "and" as a word or as a "ən" tail. </remarks>
        (string, int?) GetNumber(string word, string currency, bool isHead, string numFlags) {
            var suffix = NumSuffixRegex.Match(word).Value;
            if (suffix.Length > 0) { word = word[..^suffix.Length]; }
            var result = new List<(string Ps, int? Rating)>();
            if (word.StartsWith('-')) { result.Add(Lookup("minus", null, null, null)); word = word[1..]; }

            void ExtendNum(string num, bool first = true, bool escape = false) {
                var splits = Regex.Split(escape ? num : Num2Words.Cardinal(BigInteger.Parse(num)), "[^a-z]+");
                for (int i = 0; i < splits.Length; i++) {
                    var w = splits[i];
                    if (w != "and" || numFlags.Contains('&')) {
                        if (first && i == 0 && splits.Length > 1 && w == "one" && numFlags.Contains('a')) { result.Add(("ə", 4)); }
                        else { result.Add(Lookup(w, null, w == "point" ? -2 : null, null)); }
                    } else if (numFlags.Contains('n') && result.Count > 0) { result[^1] = (result[^1].Ps + "ən", result[^1].Rating); }
                }
            }

            if (IsDigits(word) && Ordinals.Contains(suffix)) { ExtendNum(Num2Words.Ordinal(BigInteger.Parse(word)), escape: true); }
            else if (result.Count == 0 && word.Length == 4 && !Currencies.ContainsKey(currency ?? "") && IsDigits(word)) { ExtendNum(Num2Words.Year(BigInteger.Parse(word)), escape: true); }
            else if (!isHead && !word.Contains('.')) {
                var num = word.Replace(",", "");
                if (num[0] == '0' || num.Length > 3) { foreach (var n in num) { ExtendNum($"{n}", first: false); } }
                else if (num.Length == 3 && !num.EndsWith("00")) {
                    ExtendNum(num[..1]);
                    if (num[1] == '0') { result.Add(Lookup("O", null, -2, null)); ExtendNum(num[2..], first: false); }
                    else { ExtendNum(num[1..], first: false); }
                } else { ExtendNum(num); }
            } else if (word.Count(c => c == '.') > 1 || !isHead) {
                bool first = true;
                foreach (var num in word.Replace(",", "").Split('.')) {
                    if (num.Length == 0) { }
                    else if (num[0] == '0' || (num.Length != 2 && num[1..].Any(c => c != '0'))) { foreach (var n in num) { ExtendNum($"{n}", first: false); } }
                    else { ExtendNum(num, first: first); }
                    first = false;
                }
            } else if (currency is not null && Currencies.TryGetValue(currency, out var units) && IsCurrency(word)) {
                var pairs = word.Replace(",", "").Split('.').Zip(new[] { units.Unit, units.Cent }, (num, unit) => (Num: num.Length == 0 ? default : BigInteger.Parse(num), Unit: unit)).ToList();
                if (pairs.Count > 1) {
                    if (pairs[1].Num == 0) { pairs.RemoveAt(1); }
                    else if (pairs[0].Num == 0) { pairs.RemoveAt(0); }
                }
                for (int i = 0; i < pairs.Count; i++) {
                    if (i > 0) { result.Add(Lookup("and", null, null, null)); }
                    ExtendNum($"{pairs[i].Num}", first: i == 0);
                    result.Add(BigInteger.Abs(pairs[i].Num) != 1 && pairs[i].Unit != "pence" ? StemS(pairs[i].Unit + "s", null, null, null) : Lookup(pairs[i].Unit, null, null, null));
                }
            } else {
                if (IsDigits(word)) { word = Num2Words.Cardinal(BigInteger.Parse(word)); }
                else if (!word.Contains('.')) {
                    var value = BigInteger.Parse(word.Replace(",", ""));
                    word = Ordinals.Contains(suffix) ? Num2Words.Ordinal(value) : Num2Words.Cardinal(value);
                } else {
                    word = word.Replace(",", "");
                    word = word[0] == '.' ? "point " + string.Join(" ", word[1..].Select(n => Num2Words.Cardinal(n - '0')))
                                          : Num2Words.Float(double.Parse(word, CultureInfo.InvariantCulture));
                }
                ExtendNum(word, escape: true);
            }

            if (result.Count == 0) { return (null, null); }
            var (ps, rating) = (string.Join(" ", result.Select(r => r.Ps)), result.Min(r => r.Rating));
            return suffix switch {
                "s" or "'s" => (AppendS(ps), rating),
                "ed" or "'d" => (AppendEd(ps), rating),
                "ing" => (AppendIng(ps), rating),
                _ => (ps, rating),
            };
        }

        string AppendCurrency(string ps, string currency) {
            if (string.IsNullOrEmpty(currency) || !Currencies.TryGetValue(currency, out var units)) { return ps; }
            var (plural, _) = StemS(units.Unit + "s", null, null, null);
            return string.IsNullOrEmpty(plural) ? ps : $"{ps} {plural}";
        }

        internal static bool IsDigits(string text) => text.Length > 0 && text.All(c => c is >= '0' and <= '9');
        static bool IsAlpha(string text) => text.Length > 0 && text.All(char.IsLetter);
        static char NumericIfNeeded(char c) => char.IsDigit(c) && char.GetNumericValue(c) is var n && n == (int)n ? (char)('0' + (int)n) : c;

        static readonly Regex VsRegex = new(@"^vs\.?$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant | RegexOptions.Compiled);
        static readonly Regex DoubledIngRegex = new(@"([bcdgklmnprstvxz])\1ing$|cking$", RegexOptions.Compiled);
        static readonly Regex NumSuffixRegex = new(@"[a-z']+$", RegexOptions.Compiled);
    }

    /// <summary> The slice of num2words' English that misaki uses -- cardinals up to centillion, ordinals, year phrasing, and float "point" readouts. </summary>
    /// <remarks> Numbers split into (word, value) pairs top-down, then pairwise-merge with "-"/"and"/","/" " joiners exactly like num2words' clean+merge does. </remarks>
    static class Num2Words {
        static readonly List<(BigInteger Value, string Word)> Cards = [];
        static readonly BigInteger MaxVal;
        static readonly Dictionary<string, string> Ords = new() {
            ["one"] = "first", ["two"] = "second", ["three"] = "third", ["four"] = "fourth", ["five"] = "fifth", ["six"] = "sixth",
            ["seven"] = "seventh", ["eight"] = "eighth", ["nine"] = "ninth", ["ten"] = "tenth", ["eleven"] = "eleventh", ["twelve"] = "twelfth",
        };

        static Num2Words() {
            string[] lows = ["non", "oct", "sept", "sext", "quint", "quadr", "tr", "b", "m"];
            string[] units = ["", "un", "duo", "tre", "quattuor", "quin", "sex", "sept", "octo", "novem"];
            string[] tens = ["dec", "vigint", "trigint", "quadragint", "quinquagint", "sexagint", "septuagint", "octogint", "nonagint"];
            string[] high = ["cent", .. tens.SelectMany(t => units.Select(u => u + t)).Reverse(), .. lows];
            for (int i = 0; i < high.Length; i++) { Cards.Add((BigInteger.Pow(10, 3 + 3 * (high.Length - i)), high[i] + "illion")); }
            Cards.AddRange([(1000, "thousand"), (100, "hundred"), (90, "ninety"), (80, "eighty"), (70, "seventy"), (60, "sixty"), (50, "fifty"), (40, "forty"), (30, "thirty")]);
            string[] lowWords = ["twenty", "nineteen", "eighteen", "seventeen", "sixteen", "fifteen", "fourteen", "thirteen", "twelve", "eleven", "ten",
                "nine", "eight", "seven", "six", "five", "four", "three", "two", "one", "zero"];
            for (int i = 0; i < lowWords.Length; i++) { Cards.Add((20 - i, lowWords[i])); }
            MaxVal = 1000 * Cards[0].Value;
        }

        public static string Cardinal(BigInteger value) {
            var minus = value < 0 ? "minus " : "";
            if (value < 0) { value = -value; }
            if (value >= MaxVal) { throw new OverflowException($"abs({value}) must be less than {MaxVal}."); }
            return minus + Clean(SplitNum(value)).Word;
        }

        public static string Ordinal(BigInteger value) {
            var words = Cardinal(value).Split(' ');
            var hyphenated = words[^1].Split('-');
            var last = hyphenated[^1].ToLowerInvariant();
            hyphenated[^1] = Ords.GetValueOrDefault(last) ?? (last.EndsWith('y') ? last[..^1] + "ieth" : last + "th");
            words[^1] = string.Join("-", hyphenated);
            return string.Join(" ", words);
        }

        public static string Year(BigInteger value) {
            var (high, low) = (value / 100, value % 100);
            if (high == 0 || (high % 10 == 0 && low < 10) || high >= 100) { return Cardinal(value); }
            return $"{Cardinal(high)} {(low == 0 ? "hundred" : low < 10 ? $"oh-{Cardinal(low)}" : Cardinal(low))}";
        }

        /// <summary> Speaks a decimal like num2words does with python floats -- integer part, "point", then the shortest-roundtrip digits one by one. </summary>
        /// <remarks> Integral doubles take the plain cardinal path (num2words checks int(x) == x first), so "5.0" reads just "five". </remarks>
        public static string Float(double value) {
            if (value == Math.Truncate(value)) { return Cardinal((BigInteger)value); }
            var (pre, postDigits) = Float2Tuple(value);
            var sb = new StringBuilder(Cardinal(pre));
            if (postDigits.Length > 0) { sb.Append(" point"); }
            foreach (var digit in postDigits) { sb.Append(' ').Append(Cardinal(digit - '0')); }
            return sb.ToString();
        }

        /// <summary> Precision comes from python's repr(float), so "5.0" reads "five point zero" and tiny values keep their scientific-notation digit count. </summary>
        static (BigInteger Pre, string PostDigits) Float2Tuple(double value) {
            int precision = PyFloatPrecision(value);
            var pre = (BigInteger)value;
            double post = Math.Abs(value - (double)pre) * double.Parse($"1e{precision}", CultureInfo.InvariantCulture);
            var postInt = Math.Abs(Math.Round(post) - post) < 0.01 ? (BigInteger)Math.Round(post) : (BigInteger)Math.Floor(post);
            return (pre, $"{postInt}".PadLeft(precision, '0'));
        }

        /// <summary> Digit-count analysis of the shortest-roundtrip repr, replicating how many decimals python's str(float) shows. </summary>
        static int PyFloatPrecision(double value) {
            if (value == 0) { return 1; }
            var repr = Math.Abs(value).ToString("R", CultureInfo.InvariantCulture);
            int cut = repr.IndexOf('E');
            var (mantissa, exponent) = cut < 0 ? (repr, 0) : (repr[..cut], int.Parse(repr[(cut + 1)..], CultureInfo.InvariantCulture));
            var digits = mantissa.Replace(".", "");
            int intLength = mantissa.IndexOf('.') is var dot && dot >= 0 ? dot : mantissa.Length;
            int leadingZeros = digits.Length - digits.TrimStart('0').Length;
            var (firstDigitPow, significantDigits) = (exponent + intLength - 1 - leadingZeros, digits.Trim('0').Length);
            if (firstDigitPow is >= -4 and < 16) { return Math.Max(significantDigits - 1 - firstDigitPow, 1); }
            return Math.Abs(firstDigitPow - (significantDigits - 1));
        }

        static List<object> SplitNum(BigInteger value) {
            foreach (var (elem, word) in Cards) {
                if (elem > value) { continue; }
                var (div, mod) = (BigInteger.One, BigInteger.Zero);
        if (value != 0) { div = BigInteger.DivRem(value, elem, out mod); }
                var output = new List<object>() { div == 1 ? ("one", BigInteger.One) : (object)SplitNum(div), (word, elem) };
                if (mod != 0) { output.Add(SplitNum(mod)); }
                return output;
            }
            throw new InvalidOperationException($"splitnum fell through for {value}");
        }

        /// <summary> num2words' pairwise fold -- adjacent (word, value) pairs merge left-to-right, nested lists collapse recursively until one pair remains. </summary>
        static (string Word, BigInteger Num) Clean(List<object> val) {
            while (val.Count != 1) {
                var output = new List<object>();
                if (val[0] is not List<object> && val[1] is not List<object>) {
                    output.Add(Merge(((string, BigInteger))val[0], ((string, BigInteger))val[1]));
                    if (val.Count > 2) { output.Add(val.GetRange(2, val.Count - 2)); }
                } else {
                    foreach (var elem in val) { output.Add(elem is List<object> { Count: 1 } single ? single[0] : elem is List<object> list ? (object)Clean(list) : elem); }
                }
                val = output;
            }
            return ((string, BigInteger))val[0];
        }

        static (string, BigInteger) Merge((string Text, BigInteger Num) left, (string Text, BigInteger Num) right) {
            if (left.Num == 1 && right.Num < 100) { return right; }
            if (left.Num < 100 && left.Num > right.Num) { return ($"{left.Text}-{right.Text}", left.Num + right.Num); }
            if (left.Num >= 100 && right.Num < 100) { return ($"{left.Text} and {right.Text}", left.Num + right.Num); }
            if (right.Num > left.Num) { return ($"{left.Text} {right.Text}", left.Num * right.Num); }
            return ($"{left.Text}, {right.Text}", left.Num + right.Num);
        }
    }
}
