# Oracle-generates parity fixtures for the JapaneseG2P port: misaki 0.9.4's JAG2P (the cutlet path Kokoro
# uses for ja voices) over held-out ja Wikipedia sentences plus targeted quirky formats.
#
# Fixture format (UTF-8, '\n' newlines, inside .gz; '\\', '\t', '\n', '\r' escaped as literal backslash pairs):
#   ja_heldout.tsv.gz : sentence<TAB>expected_phonemes
import gzip, io, os, random, re, sys

MISAKI = r'C:\Users\lyrco\source\_repos\KokoroSharpBinaries\models_notpushed\misaki'
PARITY = r'C:\Users\lyrco\source\_repos\KokoroSharpBinaries\models_notpushed\parity'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Tests', 'fixtures')
sys.path.insert(0, MISAKI)
from misaki import ja

rng = random.Random(42)
sentences = []
for fname in sorted(os.listdir(PARITY)):
    if not fname.startswith('wiki_ja_'): continue
    text = ' '.join(io.open(os.path.join(PARITY, fname), encoding='utf-8').read().split('\n'))
    sentences += [s.strip() for s in re.split(r'(?<=[。！？.!?…])', text) if 10 <= len(s.strip()) <= 180]
sentences = rng.sample(sorted(set(sentences)), 1600)
sentences += ['こんにちは、世界！', 'コンピューターで2024年3月14日に3.14と100円を計算します。', '「引用」と（括弧）です。',
              'すもももももももものうち。', 'ちょっと待ってください！', 'キャッシュとヴァイオリンとファックス。',
              'ｶﾀｶﾅとＡＢＣと１２３。', '約１，２３４億５６７８万円です。', 'あゝ素晴らしい。', '一九四五年八月十五日。',
              'ここ〜そこ、1〜10まで。', 'メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。']

g2p = ja.JAG2P()
def esc(s):
    return s.replace('\\', '\\\\').replace('\t', '\\t').replace('\n', '\\n').replace('\r', '\\r')

pairs, crashed = [], 0
for sent in sentences:
    try:
        ps, _ = g2p(sent)  # misaki itself crashes on 10+-digit numbers (its num2kana error string re-enters as digits)
    except AssertionError:
        crashed += 1
        continue
    pairs.append(f'{esc(sent)}\t{esc(ps)}')
print(f'{crashed} sentences skipped (oracle crash)')

os.makedirs(OUT, exist_ok=True)
buf = io.BytesIO()
with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0) as f:
    f.write(('\n'.join(pairs) + '\n').encode('utf-8'))
open(os.path.join(OUT, 'ja_heldout.tsv.gz'), 'wb').write(buf.getvalue())
print(f'{len(pairs)} fixtures, {len(buf.getvalue())} bytes -> ja_heldout.tsv.gz')
