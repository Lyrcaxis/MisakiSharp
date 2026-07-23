"""Generates KokoroSharp's Chinese G2P table (zh_g2p.tsv.gz) from jieba's + pypinyin's data + misaki's transcription rules.

Lines are `hanzi_or_phrase<TAB>pinyin syllables` for the hanzi->pinyin lookup (longest-match, resolving polyphones like pypinyin does),
`#syllable<TAB>ipa` for the pinyin->IPA lookup (pre-computed with misaki itself, so the C# side needs zero phonology logic),
and `@word<TAB>frequency` (+ `@@<TAB>total`) for jieba's word-segmentation dictionary.
Requires `pip install misaki[zh]`.
"""
import gzip, os, jieba
from pypinyin import lazy_pinyin, Style
from pypinyin.pinyin_dict import pinyin_dict
from pypinyin.phrases_dict import phrases_dict
from misaki.zh import ZHG2P

def to_pinyin(text): return lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
in_range = lambda text: all(0x4E00 <= ord(c) <= 0x9FFF for c in text)

entries, syllables = {}, set()
for cp in pinyin_dict:
    if in_range(chr(cp)): entries[chr(cp)] = to_pinyin(chr(cp))
for phrase in phrases_dict:
    if in_range(phrase): entries[phrase] = to_pinyin(phrase)
for pinyins in entries.values(): syllables.update(pinyins)

# Chars whose pinyin misaki can't transcribe would crash it too -- we just leave them out and let C# skip them.
ipa_map, failed = {}, set()
for syl in sorted(syllables):
    try: ipa_map[syl] = ZHG2P.py2ipa(syl).replace(chr(815), '')
    except Exception: failed.add(syl)
entries = { k: v for k, v in entries.items() if not any(s in failed for s in v) }

jieba.initialize()
words = { w: freq for w, freq in jieba.dt.FREQ.items() if freq > 0 and in_range(w) }

out_path = r'C:\Users\lyrco\source\repos\MisakiSharp\data\zh_g2p.tsv.gz'
with gzip.open(out_path, 'wt', encoding='utf-8', newline='\n') as f:
    for syl, ipa in sorted(ipa_map.items()): f.write(f"#{syl}\t{ipa}\n")
    f.write(f"@@\t{int(jieba.dt.total)}\n")
    for word, freq in sorted(words.items()): f.write(f"@{word}\t{freq}\n")
    for key, pinyins in sorted(entries.items()): f.write(f"{key}\t{' '.join(pinyins)}\n")

print(f"entries: {len(entries)}, syllables: {len(ipa_map)}, words: {len(words)}, failed: {len(failed)} {sorted(failed)[:10]}")
print(f"{out_path}: {os.path.getsize(out_path)/1024:.0f} KB")
