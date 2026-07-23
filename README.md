# MisakiSharp

A native C# port of [misaki](https://github.com/hexgrad/misaki) — [Kokoro TTS](https://github.com/hexgrad/kokoro)'s official G2P engine — with zero Python and zero espeak in the loop.

Every module is verified at mass scale against the reference implementation, on held-out corpora it was never tuned on:

| Module | Verified against | Parity |
|---|---|---|
| `ChineseG2P` (jieba posseg + tone sandhi + pinyin→zhuyin + cn2an) | misaki 0.9.4 `ZHG2P(version='1.1')` | **100.00%** (1936/1936 sentences) |
| `EnglishG2P` (lexicon, stress, numbers, currency, homographs) | misaki 0.9.4 `en.G2P` | 51,623 fixtures at 100% |
| `EnglishTokenizer` | spaCy `en_core_web_sm` tokenization | **100%** (56,216 lines) |
| `EnglishTagger` (2.8MB perceptron distilled from spaCy) | spaCy Penn Treebank tags | 97.5% tags → **99.20%** end-to-end phoneme parity |

## Usage

```csharp
var english = new EnglishG2P(EnglishG2P.DefaultTagger); // American English (british: true for en-GB)
var (phonemes, tokens) = english.Phonemize("On March 3rd, I read that the record was $1,234.56!");

var zhuyin = ChineseG2P.Phonemize("你好，世界！欢迎使用。"); // Kokoro v1.1 format, tone sandhi included
var arrows = ChineseG2P.PhonemizeLegacy("你好，世界！");     // Kokoro v1.0 format, for the older zh voices
```

Phoneme output uses Kokoro's phoneme alphabet. Rare out-of-lexicon English words return unk (`❓`) unless you plug an `espeakFallback`; latin spans inside Chinese text route through `ChineseG2P.EnglishPhonemizer` when set.

The building blocks are public and useful on their own: `EnglishTokenizer` (spaCy-equivalent tokenization), `EnglishTagger` (fast PTB tagging), `PossegTagger` (bit-exact jieba posseg / cut-for-search), `ToneSandhi`, and `ZhFrontend`.

## Data

The dictionaries (~22MB) embed into the assembly — pinyin/jieba tables, misaki's English lexicons, spaCy-derived tokenization rules, and the distilled tagger. Everything regenerates from `generators/*.py` against the upstream sources.

## License
- [Apache-2.0](LICENSE), like misaki itself. See [NOTICE](NOTICE) for the full lineage (PaddleSpeech, jieba, pypinyin, spaCy, num2words).
- Versioning tracks the upstream misaki release the port is verified against (currently **0.9.4**).
