# Emit MisakiSharp/data/<code>_espeak.tsv.gz, then score THAT artifact with the exact logic EspeakReplay.cs uses.
# Verifying the shipped file (not the intermediates) is the point -- it catches format drift.
#
# Reports three buckets that sum to 100%, because "parity on covered chunks" hides every failure behind a
# conditional: chunks answered correctly, chunks answered wrongly, and chunks nothing could answer.
import gzip, io, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_espeak_lts as L

sys.stdout.reconfigure(encoding='utf-8')

DUMP = r'C:\Users\lyrco\source\_repos\KokoroSharpBinaries\models_notpushed\espeak_dump'
DATA = r'C:\Users\lyrco\source\repos\MisakiSharp\data'
FIXTURES = r'C:\Users\lyrco\source\repos\MisakiSharp\Tests\fixtures'
VOICES = {'es': 'es', 'fr': 'fr-fr', 'it': 'it', 'pt': 'pt-br', 'hi': 'hi'}
FLAGS = re.compile(r'\(.+?\)')
SKIP = set('ˈˌ-^ ')
NASALS = set('nmŋɲɳ')
ZERO_WIDTH = str.maketrans('', '', '\u200b\u200c\u200d\ufeff')


def unescape(s):
    out, i = [], 0
    while i < len(s):
        if s[i] == '\\':
            i += 1
            out.append({'t': '\t', 'n': '\n', 'r': '\r'}.get(s[i], s[i]))
        else:
            out.append(s[i])
        i += 1
    return ''.join(out)


def postproc(raw):
    line = raw.replace('\r\n', '\n').replace(chr(0x361), '^').strip().replace('\n', ' ').replace('  ', ' ')
    return ' '.join(w.strip() for w in FLAGS.sub('', line).split(' ') if w.strip())


def emit(code):
    words, pairs, width = [], [], 0
    for line in gzip.open(os.path.join(DUMP, f'{code}_words6.tsv.gz'), 'rt', encoding='utf-8'):
        if line.startswith('#'): continue
        cols = line.rstrip('\n').split('\t')
        words.append(cols)
        width = max(width, len(cols) - 6)
    # A Title-case entry that behaves exactly like its lowercase form is dead weight -- lookup falls back to
    # lowercase. An all-caps entry is NOT: that fallback is deliberately withheld from acronyms, which espeak
    # spells out letter by letter, and a bare 'A' is all-caps by that test -- dropping it lost every
    # sentence-initial Portuguese article.
    rows = {cols[0]: cols[1:] for cols in words}
    words = [cols for cols in words
             if not (cols[0][:1].isupper() and not cols[0].isupper()
                     and rows.get(cols[0][:1].lower() + cols[0][1:]) == cols[1:])]
    words.sort(key=lambda cols: cols[0].encode('utf-8'))  # sorted for runtime binary search
    assert len({cols[0] for cols in words}) == len(words), f'{code} has duplicate lexicon keys'
    path = os.path.join(DUMP, f'{code}_bigrams.tsv.gz')
    if os.path.exists(path):
        for line in gzip.open(path, 'rt', encoding='utf-8'):
            if line.strip(): pairs.append((line.rstrip('\n').split('\t') + [''] * 6)[:6])
    lts = [line.rstrip('\n') for line in gzip.open(os.path.join(DUMP, f'{code}_lts.tsv.gz'), 'rt', encoding='utf-8')
           if line.strip()]
    shapes = lts[0].rsplit(' ', 1)[1]

    out = [f'# espeak-ng 1.52.0 {VOICES[code]} native replay -- word\\tfinal\\tinit\\toNasal\\toLen\\tctx\\tt0..t{width - 1}']
    out += ['\t'.join((cols + [''] * (6 + width))[:6 + width]) for cols in words]
    out.append('##BIGRAMS')
    out += ['\t'.join(p) for p in pairs]
    out.append(f'##LTS {shapes}')
    out += lts[1:]
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0) as f:
        f.write(('\n'.join(out) + '\n').encode('utf-8'))
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, f'{code}_espeak.tsv.gz'), 'wb') as f:
        f.write(buf.getvalue())
    return len(words), len(pairs), len(lts) - 1, width, len(buf.getvalue())


def load(code):
    """Mirrors EspeakReplay's constructor exactly."""
    lexicon, pairs, stress, onset, coda = {}, {}, {}, {}, {}
    shapes, segments, section = [], [], ''
    for line in gzip.open(os.path.join(DATA, f'{code}_espeak.tsv.gz'), 'rt', encoding='utf-8'):
        line = line.rstrip('\n')
        if not line: continue
        if line[0] == '#':
            if line.startswith('##BIGRAMS'): section = 'B'
            elif line.startswith('##LTS'):
                section = 'L'
                shapes = [tuple(int(n) for n in pair.split(',')) for pair in line.rsplit(' ', 1)[1].split(';')]
                segments = [{} for _ in shapes]
            continue
        c = line.split('\t')
        if section == 'B':
            pairs[(c[0], c[1])] = (c[2], c[3], c[4], c[5])
        elif section == 'L':
            if c[0] == 'S': segments[int(c[1])][c[2]] = c[3]
            elif c[0] == 'T': stress[f'{c[1]}\t{c[2]}\t{c[3]}'] = c[4]
            elif c[0] == 'O': onset[c[1]] = (c[2], c[3], int(c[4]))
            else: coda[c[1]] = c[2:]
        else:
            lexicon[c[0]] = (c[1], c[1] if c[2] == '=' else c[2], c[3], c[4], -1 if not c[5] else int(c[5]), c[6:])
    return lexicon, pairs, (shapes, segments, stress, onset, coda)


def last_phoneme(form):
    return next((c for c in reversed(form) if c not in SKIP and not (0x300 <= ord(c) <= 0x36F)), '')


def apply_head(form, delta):
    if not delta: return form
    n, _, text = delta.partition('|')
    return text + form[min(int(n), len(form)):]


def apply_tail(form, delta):
    if not delta: return form
    n, _, text = delta.partition('|')
    return form[:max(0, len(form) - int(n))] + text


def variants(token):
    """espeak phonemizes the typographic and the straight apostrophe identically, but a corpus writes one
    and a fixture the other, so half of French's elided vocabulary was unreachable over a codepoint."""
    yield token
    if "'" in token: yield token.replace("'", '\u2019')
    elif '\u2019' in token: yield token.replace('\u2019', "'")


def pieces(token):
    """Maximal runs of letters-and-marks and of decimal digits; every other character stands alone.
    'Nd' rather than str.isdigit(), which also accepts '³' and has no exact char.IsDigit counterpart."""
    import unicodedata
    out, run, kind = [], '', ''
    for char in token:
        category = unicodedata.category(char)
        current = 'L' if category[0] in 'LM' else 'N' if category == 'Nd' else ''
        if current and current == kind:
            run += char
            continue
        if run: out.append(run)
        run, kind = (char, current) if current else ('', '')
        if not current: out.append(char)
    if run: out.append(run)
    return out


def main():
    for code in sys.argv[1:] or list(VOICES):
        n_words, n_pairs, n_lts, width, size = emit(code)
        lexicon, pairs, (shapes, segments, stress, onset, coda) = load(code)
        L.SHAPES = shapes

        def found(token):
            forms = list(variants(token))
            return (next((f for f in forms if f in lexicon), None) or
                    (None if token.isupper() else
                     next((c for f in forms for c in (f[:1].lower() + f[1:], f.lower()) if c in lexicon), None)))

        def guess(token):
            emission = L.predict_segments(token, segments)
            form = L.apply_stress(emission, stress, token)
            bare = form.replace('ˈ', '').replace('ˌ', '')
            head = onset.get(bare[:2]) or onset.get(bare[:1]) or ('', '', -1)
            return (form, form, head[0], head[1], head[2], coda.get(bare[-2:]) or coda.get(bare[-1:]) or [])

        def resolve(token):
            hit = found(token)
            return lexicon[hit] if hit else guess(token if token.isupper() else token.lower())

        def tokens_of(chunk):
            out = []
            for token in chunk.split():
                core = (token.strip('।॥') if code == 'hi' else token).translate(ZERO_WIDTH)
                if not core: continue
                out += [core] if found(core) else pieces(core)
            return out

        def phonemize(tokens):
            entries = [resolve(t) for t in tokens]
            forms = [e[0] if i == len(entries) - 1 else e[1] for i, e in enumerate(entries)]
            for i in range(1, len(forms)):
                if not forms[i] or not forms[i - 1]: continue
                last = i == len(forms) - 1
                if (known := pairs.get((tokens[i - 1], tokens[i]))) is not None:
                    forms[i - 1] = apply_tail(forms[i - 1], known[2] if last else known[0])
                    forms[i] = apply_head(forms[i], known[3] if last else known[1])
                    continue
                forms[i] = apply_head(forms[i], entries[i][2] if last_phoneme(forms[i - 1]) in NASALS else entries[i][3])
                tails, ctx = entries[i - 1][5], entries[i][4]
                if 0 <= ctx < len(tails): forms[i - 1] = apply_tail(forms[i - 1], tails[ctx])
            return ' '.join(g for f in forms for g in f.split(' ') if g)

        total = covered = hit = 0
        for line in gzip.open(os.path.join(FIXTURES, f'espeak_{code}_chunks.tsv.gz'), 'rt', encoding='utf-8'):
            if not line.strip(): continue
            chunk, raw = (unescape(x) for x in line.rstrip('\n').split('\t'))
            tokens = tokens_of(chunk)
            if not tokens: continue
            total += 1
            covered += all(found(t) for t in tokens)
            hit += phonemize(tokens) == postproc(raw)
        print(f'{code}: words={n_words} pairs={n_pairs} lts={n_lts} tails={width} gz={size / 1e6:.2f} MB   '
              f'in-vocabulary {100 * covered / total:.1f}%   '
              f'correct {hit}/{total} ({100 * hit / total:.2f}%)   wrong {100 * (total - hit) / total:.2f}%')


if __name__ == '__main__':
    main()
