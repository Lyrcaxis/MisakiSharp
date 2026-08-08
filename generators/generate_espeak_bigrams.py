# Base + exceptions, the ja-matrix trick applied to word boundaries.
# Mines frequent word pairs from a Wikipedia dump prefix, asks espeak what each pair really sounds like,
# and stores ONLY the pairs the compositional model gets wrong.
#
# Pairs occurring in the chunk fixtures are EXCLUDED, so scoring on those fixtures stays honest held-out.
# Writes <out>/<code>_bigrams.tsv.gz : w1 \t w2 \t tailMedial \t headMedial \t tailFinal \t headFinal
import bz2, ctypes, gzip, io, os, re, sys, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

sys.stdout.reconfigure(encoding='utf-8')

FIXTURES = r'C:\Users\lyrco\source\repos\MisakiSharp\Tests\fixtures'
OUT = r'C:\Users\lyrco\source\_repos\KokoroSharpBinaries\models_notpushed\espeak_dump'
WIKI = {'es': 'eswiki', 'fr': 'frwiki', 'it': 'itwiki', 'pt': 'ptwiki', 'hi': 'hiwiki'}
LANGS = {'es': ('es', 'tipo'), 'fr': ('fr-fr', 'chose'), 'it': ('it', 'topo'),
         'pt': ('pt-br', 'tipo'), 'hi': ('hi', 'पता')}
TOP_BIGRAMS = int(os.environ.get('TOPB', 400000))
FLAGS = re.compile(r'\(.+?\)')
SKIP = set('ˈˌ-^ ')
NASALS = set('nmŋɲɳ')
MARKUP = re.compile(r'(?s)<ref.*?</ref>|<[^>]+>|\{\{.*?\}\}|\{\|.*?\|\}|\[\[[^]|]*\||https?://\S+|[\[\]{}|=*#:;\']')
TOKEN = re.compile(r"[^\W\d_](?:[\w'\u2019.-]*[^\W\d_])?|\d+", re.UNICODE)

_lib = _base = None
_base_n = 0


def _init(voice, base, code):
    global _lib, _base, _base_n
    load_lexicon(code)   # workers are spawned fresh on Windows, so they must load it themselves
    import espeakng_loader
    _lib = ctypes.cdll.LoadLibrary(str(espeakng_loader.get_library_path()))
    _lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    assert _lib.espeak_Initialize(0x02, 0, str(espeakng_loader.get_data_path()).encode('utf-8'), 0) > 0
    _lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
    if _lib.espeak_SetVoiceByName(voice.encode('utf-8')) != 0:
        class Voice(ctypes.Structure):
            _fields_ = [('name', ctypes.c_char_p), ('languages', ctypes.c_char_p), ('identifier', ctypes.c_char_p),
                        ('gender', ctypes.c_ubyte), ('age', ctypes.c_ubyte), ('variant', ctypes.c_ubyte),
                        ('xx1', ctypes.c_ubyte), ('score', ctypes.c_int), ('spare', ctypes.c_void_p)]
        v = Voice(languages=voice.encode('utf-8'))
        assert _lib.espeak_SetVoiceByProperties(ctypes.byref(v)) == 0, voice
    _base = base
    _base_n = len(t2p(base).split(' '))


def t2p(text):
    ptr = ctypes.c_char_p(text.encode('utf-8'))
    tp = ctypes.pointer(ctypes.cast(ptr, ctypes.c_void_p))
    out = []
    while tp.contents.value is not None:
        ph = _lib.espeak_TextToPhonemes(tp, 1, 0x02 | 0x01 << 7 | 0x361 << 8)
        if ph: out.append(ph.decode('utf-8'))
    joined = ' '.join(out).replace(chr(0x361), '^').replace('\n', ' ').strip()
    while '  ' in joined: joined = joined.replace('  ', ' ')
    return FLAGS.sub('', joined).strip()


def head_delta(base, variant):
    if base == variant: return ''
    keep = 0
    while keep < len(base) and keep < len(variant) and base[len(base) - 1 - keep] == variant[len(variant) - 1 - keep]:
        keep += 1
    return f'{len(base) - keep}|{variant[:len(variant) - keep]}'


def tail_delta(base, variant):
    if base == variant: return ''
    keep = 0
    while keep < len(base) and keep < len(variant) and base[keep] == variant[keep]:
        keep += 1
    return f'{len(base) - keep}|{variant[keep:]}'


def last_base(form):
    return next((c for c in reversed(form) if c not in SKIP and not (0x300 <= ord(c) <= 0x36F)), '')


def apply_head(form, delta):
    if not delta: return form
    n, _, text = delta.partition('|')
    return text + form[int(n):]


def apply_tail(form, delta):
    if not delta: return form
    n, _, text = delta.partition('|')
    return (form[:len(form) - int(n)] if int(n) else form) + text


LEX = {}


def load_lexicon(code):
    if LEX: return
    for line in gzip.open(os.path.join(OUT, f'{code}_words6.tsv.gz'), 'rt', encoding='utf-8'):
        if line.startswith('#'): continue
        cols = line.rstrip('\n').split('\t')
        word, final, init, on, ol, ctx = (cols + [''] * 6)[:6]
        LEX[word] = (final, final if init == '=' else init, on, ol, ctx, cols[6:])


def predict(w1, w2, w2_final):
    """What the compositional model produces for this pair in isolation."""
    f1, i1, _, _, _, tails1 = LEX[w1]
    f2, i2, on2, ol2, ctx2, _ = LEX[w2]
    a, b = i1, (f2 if w2_final else i2)
    b = apply_head(b, on2 if last_base(a) in NASALS else ol2)
    if ctx2.isdigit() and int(ctx2) < len(tails1):
        a = apply_tail(a, tails1[int(ctx2)])
    return a, b


def measure(pair):
    w1, w2 = pair
    n1, n2 = len(LEX[w1][1].split(' ')), len(LEX[w2][1].split(' '))
    rows = []
    for w2_final in (False, True):
        text = f'{w1} {w2}' if w2_final else f'{w1} {w2} {_base}'
        parts = t2p(text).split(' ')
        want = n1 + n2 + (0 if w2_final else _base_n)
        if len(parts) != want:
            rows.append(('', ''))
            continue
        actual_a, actual_b = ' '.join(parts[:n1]), ' '.join(parts[n1:n1 + n2])
        pred_a, pred_b = predict(w1, w2, w2_final)
        rows.append(('', '') if (actual_a, actual_b) == (pred_a, pred_b)
                    else (tail_delta(LEX[w1][1], actual_a),
                          head_delta(LEX[w2][0] if w2_final else LEX[w2][1], actual_b)))
    return w1, w2, rows[0], rows[1]


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


def count_bigrams(text):
    """Bigram counting is pure CPU over ~8MB blocks, so it fans out across cores like the espeak probing."""
    counts = Counter()
    for sentence in re.split(r'(?<=[.!?…।])\s+', text):
        words = TOKEN.findall(sentence)
        counts.update(zip(words, words[1:]))
    return counts


def wiki_text(code):
    path = os.path.join(OUT, f'{WIKI[code]}_prefix.bz2')
    decomp, pending, chunks = bz2.BZ2Decompressor(), b'', []
    with open(path, 'rb') as f:
        while raw := f.read(4 << 20):
            data = raw
            while data:
                try:
                    pending += decomp.decompress(data)
                except (EOFError, OSError):
                    data = b''
                    break
                if not decomp.eof: break
                data, decomp = decomp.unused_data, bz2.BZ2Decompressor()
            if len(pending) > (8 << 20):
                chunks.append(MARKUP.sub(' ', pending.decode('utf-8', 'ignore')))
                pending = b''
    if pending: chunks.append(MARKUP.sub(' ', pending.decode('utf-8', 'ignore')))
    return chunks


def main():
    for code in sys.argv[1:] or list(LANGS):
        voice, base = LANGS[code]
        LEX.clear()
        load_lexicon(code)

        # Held-out by default: pairs seen in the fixtures are withheld so scoring on them stays honest.
        # NOBAN=1 measures the CEILING instead -- how much of the residual is pair-level at all.
        banned = set()
        if not os.environ.get('NOBAN'):
            for line in gzip.open(os.path.join(FIXTURES, f'espeak_{code}_chunks.tsv.gz'), 'rt', encoding='utf-8'):
                if not line.strip(): continue
                words = unescape(line.rstrip('\n').split('\t')[0]).split()
                banned.update(zip(words, words[1:]))

        start = time.time()
        counts = Counter()
        blocks = wiki_text(code)
        with ProcessPoolExecutor(os.cpu_count()) as pool:
            for partial in pool.map(count_bigrams, blocks):
                counts.update(partial)
        pairs = [p for p, _ in counts.most_common()
                 if p not in banned and p[0] in LEX and p[1] in LEX and LEX[p[0]][1] and LEX[p[1]][1]][:TOP_BIGRAMS]
        print(f'[{code}] mined {len(counts)} bigrams in {time.time() - start:.0f}s, probing {len(pairs)}', flush=True)

        start, rows = time.time(), []
        with ProcessPoolExecutor(os.cpu_count(), initializer=_init, initargs=(voice, base, code)) as pool:
            for row in pool.map(measure, pairs, chunksize=256):
                rows.append(row)
        keep = [r for r in rows if any(r[2]) or any(r[3])]
        lines = [f'{w1}\t{w2}\t{m[0]}\t{m[1]}\t{f[0]}\t{f[1]}' for w1, w2, m, f in keep]
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0) as f:
            f.write(('\n'.join(lines) + '\n').encode('utf-8'))
        with open(os.path.join(OUT, f'{code}_bigrams.tsv.gz'), 'wb') as f:
            f.write(buf.getvalue())
        print(f'[{code}] {time.time() - start:.0f}s  exceptions={len(keep)}/{len(rows)} '
              f'({100 * len(keep) / max(1, len(rows)):.1f}%)  gz={len(buf.getvalue()) / 1e6:.2f} MB', flush=True)


if __name__ == '__main__':
    main()
