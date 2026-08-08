namespace MisakiSharp.Tests;

using System.IO.Compression;
using System.Text;

using Xunit;

/// <summary> Held-out corpora phonemized by the reference misaki 0.9.4 (see the generators' parity harnesses) versus this port, sentence by sentence. </summary>
public class ParityTests {
    static string[] ReadFixture(string name) {
        using var reader = new StreamReader(new GZipStream(File.OpenRead(Path.Combine("fixtures", name)), CompressionMode.Decompress), Encoding.UTF8);
        return reader.ReadToEnd().TrimEnd('\n').Split('\n');
    }

    [Fact]
    public void ChineseMatchesMisakiExactly() {
        var (inputs, expected) = (ReadFixture("zh_heldout.txt.gz"), ReadFixture("zh_heldout_expected.txt.gz"));
        var mismatches = inputs.Zip(expected).Count(pair => ChineseG2P.Phonemize(pair.First).Trim() != pair.Second.Trim());
        Assert.Equal(0, mismatches);
    }

    [Fact]
    public void EnglishMatchesMisakiWithinTaggerTolerance() {
        var (inputs, expected) = (ReadFixture("en_heldout.txt.gz"), ReadFixture("en_heldout_expected.txt.gz"));
        var g2p = new EnglishG2P(EnglishG2P.DefaultTagger);
        var matches = inputs.Zip(expected).Count(pair => g2p.Phonemize(pair.First).Phonemes.Trim() == pair.Second.Trim());
        Assert.True(matches >= inputs.Length * 0.99, $"parity {matches}/{inputs.Length} fell below 99%");
    }

    // The espeak fixtures also record every raw espeak response the oracle saw, so the ported pipeline replays them without spawning espeak.
    static string Unescape(string s) {
        var sb = new System.Text.StringBuilder(s.Length);
        for (int i = 0; i < s.Length; i++) {
            if (s[i] != '\\') { sb.Append(s[i]); continue; }
            sb.Append(s[++i] switch { 't' => '\t', 'n' => '\n', 'r' => '\r', var c => c });
        }
        return sb.ToString();
    }

    static Dictionary<string, string> ReadChunkMap(string name)
        => ReadFixture(name).Select(line => line.Split('\t')).ToDictionary(cols => Unescape(cols[0]), cols => Unescape(cols[1]));

    [Fact]
    public void EspeakLanguagesMatchMisakiExactly() {
        foreach (var lang in (string[])["es", "fr", "hi", "it", "pt"]) {
            var chunks = ReadChunkMap($"espeak_{lang}_chunks.tsv.gz");
            var g2p = new EspeakG2P(chunk => chunks[chunk]);
            var pairs = ReadFixture($"espeak_{lang}_heldout.tsv.gz").Select(line => line.Split('\t'));
            var mismatches = pairs.Select(pair => (Input: Unescape(pair[0]), Expected: Unescape(pair[1]))).Where(pair => g2p.Phonemize(pair.Input) != pair.Expected).ToList();
            if (mismatches is [var (input, expected), ..]) { Assert.Fail($"{lang}: {mismatches.Count} diverged from misaki, first: '{input}' -> '{g2p.Phonemize(input)}' (expected '{expected}')"); }
        }
    }

    // Held-out sentences phonemized with no espeak anywhere in the process -- every word the lexicon lacks is
    // answered by the letter-to-sound model, so there is nothing to fall back to and nothing conditional to hide behind.
    [Fact]
    public void EspeakReplayMatchesEspeakWithoutEspeak() {
        var floors = ((string Lang, double Floor)[])[("es", 0.94), ("fr", 0.82), ("hi", 0.52), ("it", 0.85), ("pt", 0.95)];
        var report = floors.Select(entry => {
            var g2p = new EspeakG2P(EspeakReplay.Provider(entry.Lang));
            var pairs = ReadFixture($"espeak_{entry.Lang}_heldout.tsv.gz").Select(line => line.Split('\t')).Select(cols => (Input: Unescape(cols[0]), Expected: Unescape(cols[1]))).ToList();
            int matches = pairs.Count(pair => g2p.Phonemize(pair.Input) == pair.Expected);
            return (entry.Lang, entry.Floor, Matches: matches, Total: pairs.Count, Rate: (double)matches / pairs.Count);
        }).ToList();
        // One assert over all five, so a regression in one language still reports the other four.
        var summary = string.Join("  ", report.Select(r => $"{r.Lang} {r.Matches}/{r.Total} ({r.Rate:P2})"));
        Assert.True(report.All(r => r.Rate >= r.Floor), $"below floor: {string.Join(", ", report.Where(r => r.Rate < r.Floor).Select(r => $"{r.Lang} {r.Rate:P2} < {r.Floor:P0}"))}  --  {summary}");
    }

    // Sentences flagged 'ok' were fully covered by the hi dump at generation time; those must replay espeak byte-exactly with no espeak involved.
    [Fact]
    public void HindiNativeMatchesEspeakOnCoveredSentences() {
        var g2p = new EspeakG2P(HindiProvider.Phonemize);
        var rows = ReadFixture("hi_native_heldout.tsv.gz").Select(line => line.Split('\t')).ToList();
        var covered = rows.Where(cols => cols[2] == "ok").Select(cols => (Input: Unescape(cols[0]), Expected: Unescape(cols[1]))).ToList();
        Assert.True(covered.Count >= rows.Count * 0.4, $"only {covered.Count}/{rows.Count} covered sentences -- lexicon regressed");
        var mismatches = covered.Where(pair => g2p.Phonemize(pair.Input) != pair.Expected).ToList();
        if (mismatches is [var (input, expected), ..]) { Assert.Fail($"hi-native: {mismatches.Count}/{covered.Count} diverged, first: '{input}' -> '{g2p.Phonemize(input)}' (expected '{expected}')"); }
    }

    [Fact]
    public void JapaneseMatchesMisakiExactly() {
        var pairs = ReadFixture("ja_heldout.tsv.gz").Select(line => line.Split('\t'));
        var mismatches = pairs.Select(pair => (Input: Unescape(pair[0]), Expected: Unescape(pair[1]))).Where(pair => JapaneseG2P.Phonemize(pair.Input) != pair.Expected).ToList();
        if (mismatches is [var (input, expected), ..]) { Assert.Fail($"ja: {mismatches.Count} diverged from misaki, first: '{input}' -> '{JapaneseG2P.Phonemize(input)}' (expected '{expected}')"); }
    }

    // The same 2914 words answered by the measured en replay instead of recorded espeak responses -- single words
    // are the fallback's whole contract, so this is the espeak-free number at exactly the seam users hit.
    [Fact]
    public void EnglishFallbackReplayMatchesEspeakWithoutEspeak() {
        var us = new EspeakFallback(british: false, EspeakReplay.Provider("en-us"));
        var gb = new EspeakFallback(british: true, EspeakReplay.Provider("en-gb"));
        var rows = ReadFixture("espeak_en_fallback.tsv.gz").Select(line => line.Split('\t')).ToList();
        int usHits = rows.Count(cols => (us.Phonemize(Unescape(cols[0])) ?? "∅") == Unescape(cols[1]));
        int gbHits = rows.Count(cols => (gb.Phonemize(Unescape(cols[0])) ?? "∅") == Unescape(cols[2]));
        var summary = $"en-us {usHits}/{rows.Count} ({(double)usHits / rows.Count:P2})  en-gb {gbHits}/{rows.Count} ({(double)gbHits / rows.Count:P2})";
        Assert.True(usHits >= rows.Count * 0.86 && gbHits >= rows.Count * 0.86, $"below floor -- {summary}");
    }

    [Fact]
    public void EnglishEspeakFallbackMatchesMisakiExactly() {
        var (us, gb) = (ReadChunkMap("espeak_en_us_chunks.tsv.gz"), ReadChunkMap("espeak_en_gb_chunks.tsv.gz"));
        var (usFallback, gbFallback) = (new EspeakFallback(british: false, chunk => us[chunk]), new EspeakFallback(british: true, chunk => gb[chunk]));
        var rows = ReadFixture("espeak_en_fallback.tsv.gz").Select(line => line.Split('\t'));
        var mismatches = rows.Count(cols => (usFallback.Phonemize(Unescape(cols[0])) ?? "∅") != Unescape(cols[1])
                                         || (gbFallback.Phonemize(Unescape(cols[0])) ?? "∅") != Unescape(cols[2]));
        Assert.Equal(0, mismatches);
    }
}
