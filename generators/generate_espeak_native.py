# Per-word espeak dump modelling a word boundary as a product of two MEASURED sides, with the
# right-hand contexts DISCOVERED rather than guessed.
#
#   left  word supplies WHAT changes  (French liaison consonant, Portuguese /s/->/z/, Spanish coda lenition)
#   right word supplies WHETHER it fires -- not derivable from its phonemes
#     (es: 'casa' vs 'kilo' are both /k/ yet only one assimilates a preceding /n/;
#      pt: voicing fires before a phonetically voiceless 'x')
#
# Pass A  probe every word after diagnostic hosts -> a signature of the effect it has leftwards.
#         Identical signatures are one context class; the frequent ones become reference contexts.
# Pass B  probe every word before one representative of each class -> its own tail delta per class,
#         plus its head deltas after a nasal / non-nasal.
#
# Writes <out>/<code>_words6.tsv.gz : word \t final \t init \t oNasal \t oLen \t ctx \t t0..t{K-1}
#        <out>/<code>_ctx.tsv       : classId \t representative \t signature \t wordCount
import ctypes, gzip, io, os, re, sys, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

sys.stdout.reconfigure(encoding='utf-8')

FIXTURES = r'C:\Users\lyrco\source\repos\MisakiSharp\Tests\fixtures'
OUT = r'C:\Users\lyrco\source\_repos\KokoroSharpBinaries\models_notpushed\espeak_dump'
# code -> (voice, wordfreq lang, baseline follower, diagnostic hosts, nasal lead, vowel lead)
LANGS = {
    'es': ('es', 'es', 'tipo', ['los', 'un', 'verdad'], 'un', 'casa'),
    'fr': ('fr-fr', 'fr', 'chose', ['les', 'un', 'petit'], 'homme', 'ami'),
    'it': ('it', 'it', 'topo', ['con', 'gas', 'ed'], 'un', 'casa'),
    'pt': ('pt-br', 'pt', 'tipo', ['os', 'bem', 'rede'], 'bem', 'casa'),
    'hi': ('hi', 'hi', 'पता', ['उस', 'मन', 'लोग'], 'मन', 'भाषा'),
}
MAX_CLASSES = 8
FLAGS = re.compile(r'\(.+?\)')

_lib = None
_base = _diags = _nasal = _vowel = None
_base_n = _diag_n = _nasal_n = _vowel_n = None
_diag_base = None
_reps = ()
_rep_n = ()


def _connect(voice):
    global _lib
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


def _init(voice, base, diags, nasal, vowel, reps=()):
    global _base, _diags, _nasal, _vowel, _base_n, _diag_n, _nasal_n, _vowel_n, _diag_base, _reps, _rep_n
    _connect(voice)
    _base, _diags, _nasal, _vowel = base, tuple(diags), nasal, vowel
    _base_n = len(t2p(base).split(' '))
    _diag_n = tuple(len(t2p(d).split(' ')) for d in _diags)
    _nasal_n, _vowel_n = len(t2p(nasal).split(' ')), len(t2p(vowel).split(' '))
    _diag_base = tuple(' '.join(t2p(f'{d} {base}').split(' ')[:n]) for d, n in zip(_diags, _diag_n))
    _reps = tuple(reps)
    _rep_n = tuple(len(t2p(r).split(' ')) for r in _reps)


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


def signature(word):
    sig = []
    for host, hn, hbase in zip(_diags, _diag_n, _diag_base):
        parts = t2p(f'{host} {word}').split(' ')
        sig.append(tail_delta(hbase, ' '.join(parts[:hn])) if len(parts) > hn else '')
    return tuple(sig)


def pass_a(word):
    return word, signature(word)


def pass_b(word):
    alone = t2p(word)
    if not alone:
        return word, '', '', '', '', ('',) * len(_reps), 'empty'
    n = len(alone.split(' '))
    parts = t2p(f'{word} {_base}').split(' ')
    if len(parts) != n + _base_n:
        return word, alone, alone, '', '', ('',) * len(_reps), 'ragged'
    init = ' '.join(parts[:n])
    tails = []
    for rep, rn in zip(_reps, _rep_n):
        p = t2p(f'{word} {rep}').split(' ')
        tails.append(tail_delta(init, ' '.join(p[:n])) if len(p) == n + rn else '')
    pn = t2p(f'{_nasal} {word}').split(' ')
    pv = t2p(f'{_vowel} {word}').split(' ')
    o_nasal = head_delta(alone, ' '.join(pn[_nasal_n:])) if len(pn) == _nasal_n + n else ''
    o_len = head_delta(alone, ' '.join(pv[_vowel_n:])) if len(pv) == _vowel_n + n else ''
    return word, alone, init, o_nasal, o_len, tuple(tails), 'ok'


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


NUMERIC = re.compile(r'^[०-९0-9]+$')
STRIP = '\u200c।॥.,;:!?()[]{}«»""\'\u2019\u201c\u201d…—–-'


def build_vocab(code, wf_lang):
    import wordfreq
    ordered, seen = [], set()

    def add(w):
        if w and w not in seen and '\t' not in w and '\n' not in w:
            seen.add(w)
            ordered.append(w)

    for w in wordfreq.top_n_list(wf_lang, 10 ** 9):
        add(w)
    # Plain 1-4 digit tokens (years, counts) are ~85-90% of every digit-bearing token in the fixtures.
    # As ordinary vocabulary entries they run through the same sandhi machinery as words, for free.
    for n in range(10000):
        add(str(n))
        if code == 'hi':
            add(str(n).translate(str.maketrans('0123456789', '०१२३४५६७८९')))
    # Sentence-initial words are capitalised, so a lowercase-only vocabulary misses the first word of every
    # sentence. Title-case variants are added here; lookup still falls back to lowercase for anything unseen.
    for w in list(ordered):
        if w[:1].isalpha() and w[:1].islower():
            add(w[:1].upper() + w[1:])
    # Fixture tokens would train the vocabulary on the test set; only add them when explicitly measuring the ceiling.
    if os.environ.get('FIXTUREVOCAB'):
        with gzip.open(os.path.join(FIXTURES, f'espeak_{code}_chunks.tsv.gz'), 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                for token in unescape(line.rstrip('\n').split('\t')[0]).split():
                    core = token.strip(STRIP)
                    if core and not NUMERIC.match(core):
                        add(core)
    corpus = os.path.join(OUT, f'{code}_corpus_words.txt')
    if os.path.exists(corpus) and not os.environ.get('NOCORPUS'):
        limit = int(os.environ.get('CORPUS_LIMIT', 10 ** 9))
        with io.open(corpus, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= limit: break
                add(line.strip())
    return ordered


def main():
    os.makedirs(OUT, exist_ok=True)
    for code in sys.argv[1:] or list(LANGS):
        voice, wf_lang, base, diags, nasal, vowel = LANGS[code]
        words = build_vocab(code, wf_lang)
        print(f'[{code}] vocab={len(words)} diags={diags}', flush=True)
        start = time.time()

        sigs = {}
        with ProcessPoolExecutor(os.cpu_count(), initializer=_init,
                                 initargs=(voice, base, diags, nasal, vowel)) as pool:
            for word, sig in pool.map(pass_a, words, chunksize=512):
                sigs[word] = sig
        counts = Counter(sigs.values())
        # the inert signature is the do-nothing class; the rest, most frequent first, become reference contexts
        ranked = [s for s, _ in counts.most_common() if any(s)][:MAX_CLASSES]
        first_word = {}
        for w in words:
            first_word.setdefault(sigs[w], w)
        reps = [first_word[s] for s in ranked]
        ctx_of = {s: i for i, s in enumerate(ranked)}
        print(f'[{code}] pass A {time.time() - start:.0f}s  signatures={len(counts)}  '
              f'reps={reps}  covered={sum(counts[s] for s in ranked)}/{len(words)}', flush=True)

        rows = []
        with ProcessPoolExecutor(os.cpu_count(), initializer=_init,
                                 initargs=(voice, base, diags, nasal, vowel, reps)) as pool:
            for row in pool.map(pass_b, words, chunksize=512):
                rows.append(row)

        lines = [f'# espeak-ng 1.52.0 {voice} -- word\\tfinal\\tinit\\toNasal\\toLen\\tctx\\tt0..t{len(reps) - 1}']
        stats, tally = Counter(), Counter()
        for word, alone, init, on, ol, tails, flag in rows:
            stats[flag] += 1
            ctx = ctx_of.get(sigs.get(word), -1)
            tally['destress'] += init != alone
            tally['heads'] += bool(on or ol)
            tally['tails'] += any(tails)
            tally[f'ctx{ctx if ctx >= 0 else "-"}'] += 1
            lines.append('\t'.join([word, alone, '=' if init == alone else init, on, ol,
                                    str(ctx) if ctx >= 0 else '', *tails]).rstrip('\t'))
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0) as f:
            f.write(('\n'.join(lines) + '\n').encode('utf-8'))
        with open(os.path.join(OUT, f'{code}_words6.tsv.gz'), 'wb') as f:
            f.write(buf.getvalue())
        with open(os.path.join(OUT, f'{code}_ctx.tsv'), 'w', encoding='utf-8', newline='\n') as f:
            for i, s in enumerate(ranked):
                f.write(f'{i}\t{reps[i]}\t{" | ".join(s)}\t{counts[s]}\n')
        print(f'[{code}] total {time.time() - start:.0f}s  {dict(stats)}  {dict(tally)}  '
              f'gz={len(buf.getvalue()) / 1e6:.2f} MB', flush=True)


if __name__ == '__main__':
    main()
