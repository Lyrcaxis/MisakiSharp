# English espeak dump for EspeakFallback's provider seam: isolated words to raw tie'd IPA, per voice.
# Unlike the five espeak languages there is no boundary model -- misaki hands the fallback single words.
#
# espeak-en treats ALL-CAPS like lowercase except a small dictionary class (IT, US), and spells out
# unpronounceable strings in either case, so every word's uppercase twin is probed but only rows that
# DIFFER from the lowercase answer ship. Letters, digits and symbols always ship: they are reached as
# sub-word pieces (W.C. -> 'W', 'C'; iPhone7 -> 'iPhone', '7') below any lexicon routing.
#
#   dump   <out>/{enus,engb}_words6.tsv.gz + {enus,engb}_heldout.tsv.gz (5k reserved, never shipped/trained)
#   emit   MisakiSharp/data/{enus,engb}_espeak.tsv.gz, then score THAT artifact with the runtime's logic
import ctypes, gzip, io, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_espeak_lts as L

sys.stdout.reconfigure(encoding='utf-8')

OUT = r'C:\Users\lyrco\source\_repos\KokoroSharpBinaries\models_notpushed\espeak_dump'
DATA = r'C:\Users\lyrco\source\repos\MisakiSharp\data'
FIXTURES = r'C:\Users\lyrco\source\repos\MisakiSharp\Tests\fixtures'
VOICES = {'enus': 'en-us', 'engb': 'en-gb'}
SYMBOLS = list("%°/&=+×–—-*~³²½§€$£¥ªº‰·±÷≈≠≤≥@#^|\\<>'’") + ['°C', '°F', '...', '…']
HELDOUT = 5000
FLAGS = re.compile(r'\(.+?\)')

_lib = None


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


def wordlist():
    import wordfreq
    base = [w for w in wordfreq.top_n_list('en', 500000, wordlist='large') if ' ' not in w and '\t' not in w]
    held = set(random.Random(42).sample(base[50000:], HELDOUT))
    caps = [w.upper() for w in base if w not in held and w.isalpha() and w == w.lower() and w.upper() != w]
    extra = [str(n) for n in range(101)] + SYMBOLS
    seen, keep = set(), []
    for w in [w for w in base if w not in held] + extra + caps:
        if w not in seen: seen.add(w); keep.append(w)
    return keep, sorted(held)


def worker(voice, infile, outfile):
    """Sacrificial probe process: espeak-gb segfaults on some ALL-CAPS strings, so results stream to disk
    line by line and the parent resumes past whatever killed us. Those tokens get no row -- espeak itself
    has no answer for them (the binary-backed pipeline would crash on the same input)."""
    _connect(voice)
    with open(outfile, 'a', encoding='utf-8', newline='\n') as out:
        for line in open(infile, encoding='utf-8'):
            word = line.rstrip('\n')
            out.write(f'{word}\t{t2p(word)}\n')
            out.flush()


def run_chunk(args):
    import subprocess
    voice, index, chunk = args
    base = os.path.join(OUT, f'_en_probe_{voice}_{index}.tsv')
    open(base, 'w').close()
    crashed = []
    while (done := sum(1 for _ in open(base, encoding='utf-8'))) < len(chunk):
        with open(base + '.in', 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(chunk[done:]) + '\n')
        r = subprocess.run([sys.executable, os.path.abspath(__file__), 'worker', voice, base + '.in', base], capture_output=True, text=True)
        if r.returncode != 0:
            assert 'Traceback' not in (r.stderr or ''), r.stderr[-800:]
            done = sum(1 for _ in open(base, encoding='utf-8'))
            with open(base, 'a', encoding='utf-8', newline='\n') as f:
                f.write(f'{chunk[done]}\t\x00\n')
            crashed.append(chunk[done])
    os.remove(base + '.in')
    return base, crashed


def probe(code, voice, words, held):
    from concurrent.futures import ThreadPoolExecutor
    todo = words + held
    jobs = [(voice, i, todo[i:i + 10000]) for i in range(0, len(todo), 10000)]
    results, skipped = {}, []
    with ThreadPoolExecutor(os.cpu_count()) as pool:
        for n, (base, crashed) in enumerate(pool.map(run_chunk, jobs)):
            skipped += crashed
            for line in open(base, encoding='utf-8'):
                w, p = line.rstrip('\n').split('\t')
                if p != '\x00': results[w] = p
            os.remove(base)
            if n % 16 == 0: print(f'[{code}] chunk {n}/{len(jobs)}', flush=True)
    for name, keys in [('words6', words), ('heldout', held)]:
        rows = [f'{w}\t{results[w]}\t=' if name == 'words6' else f'{w}\t{results[w]}' for w in keys if w in results]
        with gzip.open(os.path.join(OUT, f'{code}_{name}.tsv.gz'), 'wt', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(rows) + '\n')
    print(f'[{code}] dumped {len(results)} rows, {len(skipped)} espeak-crashers skipped {skipped[:10]}', flush=True)


def dump():
    words, held = wordlist()
    print(f'{len(words)} probes + {len(held)} held-out, per voice', flush=True)
    for code, voice in VOICES.items(): probe(code, voice, words, held)


def rules():
    """The runtime LTS only ever answers words outside the shipped lexicon, which espeak answers with its RULE
    engine, not its dictionary -- so the LTS trains on the rule engine's own outputs over generated strings."""
    lines = [l.rstrip('\n') for l in open(os.path.join(OUT, 'en_rule_wordlist.txt'), encoding='utf-8') if l.strip()]
    words, held = lines[:-20000], lines[-20000:]
    print(f'{len(words)} rule probes + {len(held)} held-out, per voice', flush=True)
    for code, voice in VOICES.items(): probe(code + 'r', voice, words, held)


def emit(code):
    rows, ipa = [], {}
    for name in ('', 'w', 'w2'):
        path = os.path.join(OUT, f'{code}{name}_words6.tsv.gz')
        if not os.path.exists(path): continue
        for line in gzip.open(path, 'rt', encoding='utf-8'):
            c = line.rstrip('\n').split('\t')
            if c[0] in ipa or not c[1]: continue
            rows.append(c)
            ipa[c[0]] = c[1]
    # No misaki-lexicon filter: a gold word never reaches the fallback WHOLE, but its row is what compounds
    # and alphanumerics resolve through (iPhone7 -> 'iphone' + '7', state-funded -> 'state' + 'funded').
    words = [c for c in rows if not (c[0].isupper() and c[0].isalpha() and ipa.get(c[0].lower()) == c[1])]
    words.sort(key=lambda c: c[0].encode('utf-8'))
    assert len({c[0] for c in words}) == len(words), f'{code} has duplicate lexicon keys'
    lts_path = next(p for p in (os.path.join(OUT, f'{code}c_lts.tsv.gz'), os.path.join(OUT, f'{code}_lts.tsv.gz'))
                    if os.path.exists(p))
    lts = [line.rstrip('\n') for line in gzip.open(lts_path, 'rt', encoding='utf-8') if line.strip()]

    out = [f'# espeak-ng 1.52.0 {VOICES[code]} native replay -- word\\tfinal\\tinit\\toNasal\\toLen\\tctx']
    out += ['\t'.join((c + [''] * 6)[:6]) for c in words]
    out.append(f'##LTS {lts[0].rsplit(" ", 1)[1]}')
    out += lts[1:]
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0) as f:
        f.write(('\n'.join(out) + '\n').encode('utf-8'))
    with open(os.path.join(DATA, f'{code}_espeak.tsv.gz'), 'wb') as f:
        f.write(buf.getvalue())
    return len(words), len(lts) - 1, len(buf.getvalue())


def verify(code):
    import generate_espeak_emit as E
    n_words, n_lts, size = emit(code)
    lexicon, _, (shapes, segments, stress, onset, coda) = E.load(code)
    L.SHAPES = shapes

    def found(token):
        forms = list(E.variants(token))
        return (next((f for f in forms if f in lexicon), None) or
                next((c for f in forms for c in (f[:1].lower() + f[1:], f.lower()) if c in lexicon), None))

    def possessive(final):
        tail = E.last_phoneme(final)
        if tail == 'ʃ': return 'ɪz'
        if tail in 'szʒ': return 'ɪz' if code == 'engb' else 'ᵻz'
        return 's' if tail in 'ptkfx' else 'z'

    def resolve(token):
        hit = found(token)
        if hit: return lexicon[hit][0]
        if len(token) > 2 and token[-1] == 's' and token[-2] in "'’":
            base = resolve(token[:-2])
            return base + possessive(base)
        emission = L.predict_segments(token.lower(), segments)
        return L.apply_stress(emission, stress, token.lower())

    VOWELS = 'aeiouyɛɔəɐɨʊɪøœɑæʌɜɤɯɵʉɶɒ'
    REDUCED = {'in': 'ɪn', 'by': 'ba^ɪ', 'to': 'tə', 'a': 'ɐ', 'up': 'ˌʌp',
               'on': 'ˌɒn' if code == 'engb' else 'ˌɔn', 'under': 'ˌʌndə' if code == 'engb' else 'ˌʌndɚ'}

    def linked(form, token):
        if form.endswith('^ɹ'): return form[:-2] + 'ɹ'
        if code == 'enus' and form.endswith('ɚ'): return form + 'ɹ'
        if code == 'engb' and token.lower().endswith('r'): return form + 'ɹ'
        if code == 'enus' and form.endswith('t'): return form[:-1] + 'ɾ'
        return form

    def phonemize(chunk):
        out = []
        for token in chunk.split():
            whole = found(token) or (len(token) > 2 and token[-1] == 's' and token[-2] in "'’")
            pieces = [token] if whole else E.pieces(token)
            forms = [resolve(p) for p in pieces]
            for j in range(2, len(pieces)):
                if pieces[j - 1] != '-': continue
                if pieces[j - 2].lower() in REDUCED: forms[j - 2] = REDUCED[pieces[j - 2].lower()]
                head = forms[j].lstrip('ˈˌ')
                if head[:1] in VOWELS and forms[j - 2]: forms[j - 2] = linked(forms[j - 2], pieces[j - 2])
            merged = []
            for j, p in enumerate(pieces):
                form = ' '.join(g for g in forms[j].split(' ') if g)
                if not form: continue
                if j > 1 and pieces[j - 1] == '-' and merged: merged[-1] += form
                else: merged.append(form)
            out += merged
        return ' '.join(out)

    total = hit = covered = 0
    fixture = {'enus': 'espeak_en_us_chunks.tsv.gz', 'engb': 'espeak_en_gb_chunks.tsv.gz'}[code]
    for line in gzip.open(os.path.join(FIXTURES, fixture), 'rt', encoding='utf-8'):
        if not line.strip(): continue
        chunk, raw = (E.unescape(x) for x in line.rstrip('\n').split('\t'))
        total += 1
        covered += all(found(t) for t in chunk.split())
        hit += phonemize(chunk) == E.postproc(raw)
    scores = []
    for name in ('', 'r'):
        path = os.path.join(OUT, f'{code}{name}_heldout.tsv.gz')
        guessed = guessed_hit = 0
        if os.path.exists(path):
            for line in gzip.open(path, 'rt', encoding='utf-8'):
                word, gold = line.rstrip('\n').split('\t')
                guessed += 1
                guessed_hit += phonemize(word) == E.postproc(gold)
        scores.append(f'{guessed_hit}/{guessed} ({100 * guessed_hit / max(1, guessed):.2f}%)')
    print(f'{code}: words={n_words} lts={n_lts} gz={size / 1e6:.2f} MB   '
          f'fixture in-vocabulary {100 * covered / total:.1f}%   correct {hit}/{total} ({100 * hit / total:.2f}%)   '
          f'held-out real {scores[0]}   held-out rules {scores[1]}', flush=True)


def main():
    if sys.argv[1] == 'worker': worker(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == 'dump': dump()
    elif sys.argv[1] == 'rules': rules()
    else:
        for code in sys.argv[2:] or list(VOICES): verify(code)


if __name__ == '__main__':
    main()
