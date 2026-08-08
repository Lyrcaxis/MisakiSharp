# Trains the letter-to-sound model that lets EspeakReplay answer for words the lexicon has never seen,
# and writes <out>/<code>_lts.tsv.gz for generate_espeak_emit.py to fold into the shipped artifact.
#
# Two jobs, deliberately kept apart -- conflating them is what held an earlier probe to 48% on Spanish:
#   segments  a letter's phonemes from its surrounding letters. Windows are asymmetric (a 'c' needs its
#             RIGHT neighbour, a final 's' its LEFT) and nested, so a wide window can be dropped whenever
#             the narrower one inside it already predicts the same thing -- exact, and it removes ~97% of them.
#   stress    a word-level property counted in syllables from the END, which no fixed letter window can see.
#             Keyed on nucleus count, which nucleus carries the written accent, and the suffix.
#
# Alignment is Viterbi-EM (a letter emits 0..3 phoneme units). Uniform initialisation is degenerate -- every
# path costs the same, since the cost depends only on how many letters there are -- so it is seeded from the
# words that already line up one letter to one unit.
import gzip, io, math, os, sys, time, unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

sys.stdout.reconfigure(encoding='utf-8')

OUT = r'C:\Users\lyrco\source\_repos\KokoroSharpBinaries\models_notpushed\espeak_dump'
LANGS = ['es', 'fr', 'it', 'pt', 'hi']
EM_ROUNDS = 4
MAX_UNITS = 3
CONTEXT = 7
SIDE = 4
STRESS = 'ˈˌ'
MODIFIERS = 'ːʰʲʷʼ˺'
VOWELS = set('aeiouyɛɔəɐɨʊɪøœɑæʌɜɤɯɵʉɶɒ')
ACCENTED = set('áéíóúàèìòùâêîôûãõĩũ')
NONE = '.'  # a nucleus with no stress mark, so a pattern stays one character per nucleus

SHAPES = sorted([(l, r) for k in range(SIDE + 1) for l, r in ((k, k), (k, k + 1)) if l + r <= CONTEXT and r <= SIDE],
                key=lambda s: (-(s[0] + s[1]), abs(s[0] - s[1])))


def is_word(word):
    """`isalpha()` is False for a Devanagari matra or virama (categories Mc/Mn), which quietly restricted
    Hindi to bare-consonant words; letters-and-marks is the test that spans both scripts."""
    return bool(word) and all(unicodedata.category(c)[0] in 'LM' for c in word)


def phoneme_units(text):
    """One unit per segment: leading stress marks, the base character, whatever binds to it (combining
    tilde, length, aspiration) and an optional '^'-tied second segment."""
    units, i, n = [], 0, len(text)
    while i < n:
        j = i
        while j < n and text[j] in STRESS: j += 1
        if j >= n: break
        j += 1
        while j < n and (0x300 <= ord(text[j]) <= 0x36F or text[j] in MODIFIERS): j += 1
        if j < n and text[j] == '^':
            j += 2
            while j < n and (0x300 <= ord(text[j]) <= 0x36F or text[j] in MODIFIERS): j += 1
        units.append(text[i:j])
        i = j
    return units


def split_stress(unit):
    k = 0
    while k < len(unit) and unit[k] in STRESS: k += 1
    return unit[:k], unit[k:]


def is_nucleus(bare): return bool(bare) and bare[0] in VOWELS


def align(letters, units, prob):
    n, m, NEG = len(letters), len(units), float('-inf')
    best = [[NEG] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for i, letter in enumerate(letters):
        row, nxt = best[i], best[i + 1]
        for k in range(m + 1):
            if row[k] == NEG: continue
            for take in range(MAX_UNITS + 1):
                if k + take > m: break
                score = row[k] + prob.get((letter, ''.join(units[k:k + take])), -12.0)
                if score > nxt[k + take]:
                    nxt[k + take] = score
                    back[i + 1][k + take] = take
    if best[n][m] == NEG: return None
    out, k = [], m
    for i in range(n, 0, -1):
        take = back[i][k]
        out.append(''.join(units[k - take:k]))
        k -= take
    return out[::-1]


_PROB = {}


def _init(prob):
    global _PROB
    _PROB = prob


def align_one(pair): return align(pair[0], pair[1], _PROB)


def train_alignment(pairs):
    counts, totals = Counter(), Counter()
    for letters, units in pairs:
        if len(letters) != len(units): continue
        for letter, unit in zip(letters, units):
            counts[(letter, unit)] += 1
            totals[letter] += 1
    prob = {key: math.log(c / totals[key[0]]) for key, c in counts.items()}
    for _ in range(EM_ROUNDS + 1):
        with ProcessPoolExecutor(os.cpu_count(), initializer=_init, initargs=(dict(prob),)) as pool:
            alignments = list(pool.map(align_one, pairs, chunksize=512))
        counts, totals = Counter(), Counter()
        for (letters, _), emission in zip(pairs, alignments):
            if emission is None: continue
            for letter, chunk in zip(letters, emission):
                counts[(letter, chunk)] += 1
                totals[letter] += 1
        prob = {key: math.log(c / totals[key[0]]) for key, c in counts.items()}
    return alignments


def build_segments(words, alignments):
    tables = [defaultdict(Counter) for _ in SHAPES]
    for word, emission in zip(words, alignments):
        if emission is None: continue
        padded = '#' * SIDE + word + '#' * SIDE
        for i, chunk in enumerate(emission):
            centre = i + SIDE
            for table, (left, right) in zip(tables, SHAPES):
                table[padded[centre - left:centre + right + 1]][chunk] += 1
    resolved = [{key: hits.most_common(1)[0][0] for key, hits in table.items()} for table in tables]
    # A wide window earns its place only by contradicting the narrower one inside it. Walking narrow to wide
    # means every table a key falls back to is already final, so this cannot change a single prediction.
    for i in range(len(SHAPES) - 2, -1, -1):
        left, kept = SHAPES[i][0], {}
        for key, chunk in resolved[i].items():
            for j in range(i + 1, len(SHAPES)):
                fallback = resolved[j].get(key[left - SHAPES[j][0]:left + 1 + SHAPES[j][1]])
                if fallback is not None:
                    if fallback != chunk: kept[key] = chunk
                    break
            else:
                kept[key] = chunk
        resolved[i] = kept
    return resolved


def predict_segments(word, tables):
    padded = '#' * SIDE + word + '#' * SIDE
    out = []
    for i in range(len(word)):
        centre = i + SIDE
        out.append(next((chunk for table, (left, right) in zip(tables, SHAPES)
                         if (chunk := table.get(padded[centre - left:centre + right + 1])) is not None), ''))
    return out


def stress_features(word, emission):
    nuclei, accent = 0, -1
    for letter, chunk in zip(word, emission):
        for unit in phoneme_units(chunk):
            if is_nucleus(split_stress(unit)[1]):
                nuclei += 1
                if letter in ACCENTED: accent = nuclei
    return nuclei, (nuclei - accent if accent >= 0 else -1)


def stress_keys(nuclei, accent, word):
    for suffix in (3, 2, 1): yield f'{nuclei}\t{accent}\t{word[-suffix:]}'
    yield f'{nuclei}\t{accent}\t'
    yield f'-1\t{accent}\t'
    yield '-1\t-1\t'


def build_stress(words, alignments, golds):
    tables = defaultdict(Counter)
    for word, emission, gold in zip(words, alignments, golds):
        if emission is None: continue
        pattern = ''.join(mark or NONE for mark, bare in map(split_stress, phoneme_units(gold)) if is_nucleus(bare))
        nuclei, accent = stress_features(word, emission)
        for key in stress_keys(nuclei, accent, word): tables[key][pattern] += 1
    resolved = {key: hits.most_common(1)[0][0] for key, hits in tables.items()}
    # Same exact pruning as the segment tables: a specific key is dead weight when the general key it would
    # fall back to already answers the same. Every fallback is strictly less specific, so ordering by
    # specificity means each is final before anything that consults it is examined.
    kept = {}
    for key in sorted(resolved, key=lambda k: (specificity(k), len(k.split('\t', 2)[2]))):
        pattern = resolved[key]
        fallback = next((kept[f] for f in stress_fallbacks(key) if f in kept), None)
        if fallback != pattern: kept[key] = pattern
    return kept


def specificity(key):
    nuclei, accent, _ = key.split('\t', 2)
    return 0 if nuclei == accent == '-1' else 1 if nuclei == '-1' else 2


def stress_fallbacks(key):
    nuclei, accent, suffix = key.split('\t', 2)
    for cut in range(1, len(suffix)): yield f'{nuclei}\t{accent}\t{suffix[cut:]}'
    if suffix: yield f'{nuclei}\t{accent}\t'
    if nuclei != '-1': yield f'-1\t{accent}\t'
    if accent != '-1': yield '-1\t-1\t'


def apply_stress(emission, table, word):
    nuclei, accent = stress_features(word, emission)
    pattern = next((table[key] for key in stress_keys(nuclei, accent, word) if key in table), '')
    out, index = [], 0
    for chunk in emission:
        for unit in phoneme_units(chunk):
            bare = split_stress(unit)[1]
            if not is_nucleus(bare):
                out.append(bare)
                continue
            mark = pattern[index] if index < len(pattern) else NONE
            out.append(('' if mark == NONE else mark) + bare)
            index += 1
    return ''.join(out)


def sandhi_tables(rows):
    """A word's onset decides what it does to the word before it; its coda decides what happens to its own
    tail. An unknown word given neither silently loses Spanish lenition and Portuguese /s/ voicing."""
    onset, coda = defaultdict(Counter), defaultdict(Counter)
    for final, on, ol, ctx, tails in rows:
        bare = final.replace('ˈ', '').replace('ˌ', '')
        if not bare: continue
        for key in {bare[:1], bare[:2]}: onset[key][(on, ol, ctx)] += 1
        for key in {bare[-1:], bare[-2:]}: coda[key][tuple(tails)] += 1
    return ({k: c.most_common(1)[0][0] for k, c in onset.items()},
            {k: c.most_common(1)[0][0] for k, c in coda.items()})


def read_dump(code):
    trainable, sandhi = [], []
    for line in gzip.open(os.path.join(OUT, f'{code}_words6.tsv.gz'), 'rt', encoding='utf-8'):
        if line.startswith('#'): continue
        c = line.rstrip('\n').split('\t')
        c += [''] * max(0, 6 - len(c))
        sandhi.append((c[1], c[3], c[4], -1 if not c[5] else int(c[5]), c[6:]))
        if c[1] and ' ' not in c[1] and len(c[0]) <= 24 and c[0] == c[0].lower() and is_word(c[0]):
            trainable.append((c[0], c[1]))
    return trainable, sandhi


def main():
    for code in sys.argv[1:] or LANGS:
        start = time.time()
        trainable, sandhi = read_dump(code)
        words = [w for w, _ in trainable]
        golds = [p for _, p in trainable]
        alignments = train_alignment([(list(w), [split_stress(u)[1] for u in phoneme_units(p)])
                                      for w, p in trainable])
        segments = build_segments(words, alignments)
        stress = build_stress(words, alignments, golds)
        onset, coda = sandhi_tables(sandhi)

        lines = [f'# letter-to-sound for out-of-lexicon words -- shapes {";".join(f"{l},{r}" for l, r in SHAPES)}']
        lines += [f'S\t{i}\t{key}\t{chunk}' for i, table in enumerate(segments) for key, chunk in table.items()]
        lines += [f'T\t{key}\t{pattern}' for key, pattern in stress.items()]
        lines += [f'O\t{key}\t{on}\t{ol}\t{ctx}' for key, (on, ol, ctx) in onset.items()]
        lines += ['\t'.join(['C', key, *tails]) for key, tails in coda.items()]
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0) as f:
            f.write(('\n'.join(lines) + '\n').encode('utf-8'))
        with open(os.path.join(OUT, f'{code}_lts.tsv.gz'), 'wb') as f:
            f.write(buf.getvalue())

        exact = sum(apply_stress(predict_segments(w, segments), stress, w) == g for w, g in trainable[:20000])
        print(f'[{code}] trained on {len(words)} words  segments={sum(len(t) for t in segments)} '
              f'stress={len(stress)} onset={len(onset)} coda={len(coda)}  '
              f'recall {100 * exact / min(20000, len(trainable)):.1f}%  '
              f'gz={len(buf.getvalue()) / 1e6:.2f} MB  {time.time() - start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
