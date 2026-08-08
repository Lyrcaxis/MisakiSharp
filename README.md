[![NuGet](https://img.shields.io/nuget/v/MisakiSharp.svg)](https://www.nuget.org/packages/MisakiSharp/)
[![NuGet Downloads](https://img.shields.io/nuget/dt/MisakiSharp.svg)](https://www.nuget.org/packages/MisakiSharp/)

# MisakiSharp

A native C# port of [hexgrad's misaki](https://github.com/hexgrad/misaki): [Kokoro TTS](https://github.com/hexgrad/kokoro)'s official G2P engine.

| Module | Verified against | Parity |
|---|---|---|
| `ChineseG2P` (jieba posseg + tone sandhi + pinyin→zhuyin + cn2an) | misaki 0.9.4 `ZHG2P(version='1.1')` | **100.00%** (1936/1936 sentences) |
| `EnglishG2P` (lexicon, stress, numbers, currency, homographs) | misaki 0.9.4 `en.G2P` | 51,623 fixtures at 100% |
| `EnglishTokenizer` | spaCy `en_core_web_sm` tokenization | **100%** (56,216 lines) |
| `EnglishTagger` (2.8MB perceptron distilled from spaCy) | spaCy Penn Treebank tags | 97.5% tags → **99.20%** end-to-end phoneme parity |
| `JapaneseG2P` (unidic lattice + cutlet kana→IPA + num2kana) | misaki 0.9.4 `JAG2P()` | **100.00%** (1593/1593 sentences) |
| `JapaneseTagger` (MeCab-identical viterbi over unidic 3.1.0) | fugashi (misaki[ja]'s tagger) | **100.00%** (14,342/14,342 parses) |
| `JapaneseTagger`'s connection costs (unidic's 481MB matrix compressed to 10MB) | exact matrix | 99.91% on held-out text |
| `EspeakG2P` (es, fr-fr, hi, it, pt-br) | misaki 0.9.4 `espeak.EspeakG2P` | **100.00%** (8000/8000 sentences, 5 languages) |
| `EspeakFallback` (English out-of-lexicon words) | misaki 0.9.4 `espeak.EspeakFallback` | **100.00%** (2914 words × en-US/en-GB) |
| `EspeakReplay` (espeak-free es, fr-fr, it, pt-br, hi) | espeak-ng 1.52.0, held-out sentences | pt 96.2% · es 95.5% · it 85.8% · fr 83.6% · hi 53.6% (whole corpus, nothing excluded) |
| `EspeakReplay` (espeak-free en-us/en-gb OOV fallback) | espeak-ng 1.52.0, misaki's 2914 OOV words | us 88.1% · gb 88.1% (via `EspeakFallback`, whole fixture) |
| `HindiProvider` (espeak-free hi, superseded by `EspeakReplay`) | espeak-ng 1.52.0 | **100.00%** on lexicon-covered sentences (~54% full coverage) |

## Usage

```csharp
var englishG2P = new EnglishG2P(british: false);
var (phonemes, tokens) = englishG2P.Phonemize("On March 3rd, I read that the record was $1,234.56!");

// Japanese and Chinese
var jPhonemes = JapaneseG2P.Phonemize("日本語は面白い！");
var cPhonemes = ChineseG2P.Phonemize("你好，世界！欢迎使用。");

// it's called EspeakG2P but it's distilled, there's no actual espeak in the backend
var langs = EspeakG2P.Languages; // `["es", "fr-fr", "it", "pt-br", "hi"]`
var spanish = new EspeakG2P("es").Phonemize("El 8% a 25 °C, según el zurbicalismo.");
```

## License
- [Apache-2.0](LICENSE), like misaki itself. See [NOTICE](NOTICE) for the full lineage (PaddleSpeech, jieba, pypinyin, spaCy, num2words).
- Versioning tracks the upstream misaki release the port is verified against (currently **0.9.4**).
