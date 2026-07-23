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
}
