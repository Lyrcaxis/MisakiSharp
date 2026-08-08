# Builds the espeak-free Hindi lexicon + verifies the native provider design against live espeak, chunk by chunk.
#
# The native Hindi path replaces only EspeakG2P's *provider*: given a punctuation-free chunk, produce the raw
# tie'd-IPA string espeak would have produced. Everything downstream (punctuation restore, e2m, flag removal)
# stays the shared, already-parity-gated pipeline. This generator:
#   1. assembles a vocabulary (wordfreq + fixture-corpus tokens + a wiki harvest + hiwiki article titles)
#   2. probes each word's phonemes; the top slice also in 3 clause positions (alone / non-final / final) --
#      espeak destresses flagged function words positionally ('mein' keeps stress clause-finally, 'ki' never does)
#   3. derives the Indian-system number verbalization from empirical slot tables (standalone / after-hundred /
#      group-before-scale forms all differ!) and verifies exhaustively against live espeak
#   4. prototypes the provider in python and gates it on a FRESH held-out wiki corpus (disjoint article set),
#      comparing postprocessed+flag-stripped chunk outputs against live espeak
#   5. writes data/hi_espeak_lexicon.tsv.gz + Tests/fixtures/hi_native_heldout.tsv.gz
#
# Espeak quirks baked in: danda attached to a word is silently dropped (no clause break); a STANDALONE danda
# token is verbalized ('purnaviram') only when nothing spoken precedes it in the chunk.
# Lexicon format (UTF-8, gz): word<TAB>alone[<TAB>nonfinal<TAB>final] -- cols 3/4 only when differing.
# '@@N*'-prefixed lines carry the number slot tables. Ties stored as '^' (pipeline-native form).
import ctypes, gzip, io, json, os, random, re, sys, time, urllib.parse, urllib.request
from collections import Counter

import espeakng_loader
import wordfreq

FIXTURES = r'C:\Users\lyrco\source\repos\MisakiSharp\Tests\fixtures'
DATA = r'C:\Users\lyrco\source\repos\MisakiSharp\data'
PARITY = r'C:\Users\lyrco\source\_repos\KokoroSharpBinaries\models_notpushed\parity'
CARRIER = 'भारत'
POSITIONAL_TOP = 6000
report = {}

# ---- espeak via the API (the byte-exact path) ----
lib = ctypes.cdll.LoadLibrary(str(espeakng_loader.get_library_path()))
lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
assert lib.espeak_Initialize(0x02, 0, str(espeakng_loader.get_data_path()).encode('utf-8'), 0) > 0
lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
assert lib.espeak_SetVoiceByName(b'hi') == 0

def t2p(text):
    ptr = ctypes.c_char_p(text.encode('utf-8'))
    tp = ctypes.pointer(ctypes.cast(ptr, ctypes.c_void_p))
    out = []
    while tp.contents.value is not None:
        ph = lib.espeak_TextToPhonemes(tp, 1, 0x02 | 0x01 << 7 | 0x361 << 8)
        if ph: out.append(ph.decode('utf-8'))
    joined = ' '.join(out).replace(chr(0x361), '^').replace('\n', ' ').strip()
    while '  ' in joined: joined = joined.replace('  ', ' ')
    return joined

CARRIER_PS = t2p(CARRIER)

# ---- corpora ----
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

def esc(s):
    return s.replace('\\', '\\\\').replace('\t', '\\t').replace('\n', '\\n').replace('\r', '\\r')

def fetch_wiki(title):
    cache = os.path.join(PARITY, f'wiki_hi_{re.sub(r"[^\w]+", "_", title)}.txt')
    if os.path.exists(cache):
        return io.open(cache, encoding='utf-8').read()
    url = (f'https://hi.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&format=json'
           f'&redirects=1&titles={urllib.parse.quote(title)}')
    req = urllib.request.Request(url, headers={'User-Agent': 'MisakiSharpFixtures/1.0 (https://github.com/Lyrcaxis/MisakiSharp; lyrcaxis@gmail.com)'})
    for attempt in range(6):
        try:
            pages = json.loads(urllib.request.urlopen(req, timeout=60).read())['query']['pages']
            break
        except urllib.error.HTTPError as e:
            if e.code != 429: raise
            time.sleep(5 * (attempt + 1))
    text = next(iter(pages.values())).get('extract', '')
    assert text, f'empty extract for hi:{title}'
    io.open(cache, 'w', encoding='utf-8', newline='\n').write(text)
    time.sleep(1)
    return text

def fetch_titles():
    cache = os.path.join(PARITY, 'hiwiki-all-titles-ns0.gz')
    if not os.path.exists(cache):
        url = 'https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-all-titles-in-ns0.gz'
        req = urllib.request.Request(url, headers={'User-Agent': 'MisakiSharpFixtures/1.0 (lyrcaxis@gmail.com)'})
        with open(cache, 'wb') as f:
            f.write(urllib.request.urlopen(req, timeout=300).read())
    with gzip.open(cache, 'rt', encoding='utf-8', errors='ignore') as f:
        return f.read().split('\n')

VOCAB_TITLES = ['मुम्बई', 'कोलकाता', 'क्रिकेट', 'बॉलीवुड', 'रामायण', 'महाभारत', 'हिमालय', 'गंगा नदी', 'अर्थशास्त्र', 'जीव विज्ञान',
                'कम्प्यूटर', 'रेल परिवहन', 'भारतीय संविधान', 'योग', 'आयुर्वेद', 'सचिन तेंदुलकर', 'रवीन्द्रनाथ ठाकुर', 'भारतीय स्वतन्त्रता आन्दोलन']
HELDOUT_TITLES = ['जयपुर', 'वायुमण्डल', 'प्रकाश', 'लोकतन्त्र', 'पुस्तकालय', 'शिक्षा', 'स्वास्थ्य', 'कृषि', 'ऊर्जा', 'संचार', 'पर्यावरण', 'खगोल शास्त्र']

def sentences_from(titles, count):
    lines = []
    for title in titles:
        text = fetch_wiki(title)
        lines += [l for l in text.split('\n') if l.strip() and not re.fullmatch(r'=+ .+? =+', l.strip())]
    sentences, seen = [], set()
    for sent in re.split(r'(?<=[.!?…।])\s+', ' '.join(lines)):
        sent = re.sub(r'\s+', ' ', sent).strip()
        if 20 <= len(sent) <= 180 and re.search(r'\w', sent) and sent not in seen:
            seen.add(sent)
            sentences.append(sent)
    rng = random.Random(42)
    return rng.sample(sentences, min(count, len(sentences)))

def tokens_of(text):
    for token in text.split():
        core = (token.strip('।॥') or token).replace('‌', '')
        if core and not re.fullmatch(r'[०-९0-9]+', core):
            yield core

fixture_chunks = {}
with gzip.open(os.path.join(FIXTURES, 'espeak_hi_chunks.tsv.gz'), 'rt', encoding='utf-8') as f:
    for line in f:
        c, r = line.rstrip('\n').split('\t')
        fixture_chunks[unescape(c)] = unescape(r)

vocab = Counter()
for chunk in fixture_chunks: vocab.update(tokens_of(chunk))
for sent in sentences_from(VOCAB_TITLES, 10 ** 9): vocab.update(tokens_of(sent))
corpus_words = set(vocab)
for w in wordfreq.top_n_list('hi', 10 ** 9): vocab[w] += 0 if w in vocab else 1
title_tokens = Counter()
for title in fetch_titles():
    for token in title.replace('_', ' ').split():
        token = token.strip('.,()"\'-')
        if token and re.fullmatch(r'[\u0900-\u097F]+', token) and not re.fullmatch(r'[०-९]+', token):
            title_tokens[token] += 1
for w, n in title_tokens.most_common(200000):
    vocab[w] += 0 if w in vocab else 1
for w in wordfreq.top_n_list('en', 15000):
    vocab[w] += 0 if w in vocab else 1
vocab.update(['*', '+', '=', '%', '@', '&', '#', '°', '।', '॥', '•', '·', '§', '©', '®', '™', '±', '×', '÷', '′', '″', '½', '¼', '¾', '₹'])
words = [w for w, _ in vocab.most_common(260000) if '\t' not in w and '\n' not in w]
report['vocab'] = len(words)

# ---- lexicon dump: alone for everything, positional forms for the frequent slice + corpus words ----
lexicon, weird = {}, 0
positional_set = set(words[:POSITIONAL_TOP]) | corpus_words
for w in words:
    alone = t2p(w)
    if w in positional_set:
        pre, post = t2p(f'{w} {CARRIER}'), t2p(f'{CARRIER} {w}')
        if pre.endswith(f' {CARRIER_PS}') and post.startswith(f'{CARRIER_PS} '):
            lexicon[w] = (alone, pre[:-len(CARRIER_PS) - 1], post[len(CARRIER_PS) + 1:])
        else:
            weird += 1
            lexicon[w] = (alone, alone, alone)
    else:
        lexicon[w] = (alone, alone, alone)
report['weird_words'] = weird
report['positional_words'] = sum(1 for a, n, f in lexicon.values() if n != a or f != a)

# ---- numbers: empirical slot tables (espeak differentiates standalone / after-hundred / group / final-tail forms) ----
NUM = {n: t2p(str(n)) for n in range(100)}
HUNDRED_FUSED = {d: t2p(str(d * 100)) for d in range(1, 10)}
SCALE_PS = {3: t2p('1000').split(' ', 1)[1], 5: t2p('100000').split(' ', 1)[1], 7: t2p('10000000').split(' ', 1)[1],
            9: t2p('1000000000').split(' ', 1)[1], 11: t2p('100000000000').split(' ', 1)[1]}
TAIL_AFTER_HUNDRED = {n: t2p(str(100 + n))[len(HUNDRED_FUSED[1]) + 1:] for n in range(1, 100)}
TAIL_FINAL = {n: t2p(str(1000 + n)).split(' ', 2)[2] for n in range(1, 100)}
GROUP = {n: t2p(str(n * 1000)).rsplit(' ', 1)[0] for n in range(1, 100)}
DIGITWISE = {d: t2p(str(d)) for d in range(10)}

def final_group(n, has_higher):
    if n == 0: return ''
    if n < 100: return TAIL_FINAL[n] if has_higher else NUM[n]
    return HUNDRED_FUSED[n // 100] + (f' {TAIL_AFTER_HUNDRED[n % 100]}' if n % 100 else '')

def number_phonemes(digits):
    digits = digits.translate(str.maketrans('०१२३४५६७८९', '0123456789'))
    if digits[0] == '0' or len(digits) > 13:
        if len(digits) == 1: return DIGITWISE[int(digits)]
        if len(digits) == 2: return f'{DIGITWISE[int(digits[0])]} {DIGITWISE[int(digits[1])]}'
        joined = ''.join(DIGITWISE[int(d)] for d in digits)
        return joined if len(digits) != 3 else f'{DIGITWISE[int(digits[0])] + DIGITWISE[int(digits[1])]} {DIGITWISE[int(digits[2])]}'
    n = int(digits)
    parts = []
    for power in (11, 9, 7, 5, 3):
        group = (n // 10 ** power) % 100 if power < 11 else n // 10 ** power
        if group: parts.append(f'{GROUP[group] if group < 100 else final_group(group, True)} {SCALE_PS[power]}')
    if n % 1000: parts.append(final_group(n % 1000, n >= 1000))
    return ' '.join(parts)

rng = random.Random(42)
num_samples = [str(n) for n in range(10000)] + [str(rng.randrange(10 ** k, 10 ** (k + 1))) for k in range(4, 13) for _ in range(300)]
num_samples += ['0' + str(rng.randrange(10 ** rng.randrange(0, 5))) for _ in range(300)] + ['0' * rng.randrange(1, 4) + str(rng.randrange(10)) for _ in range(100)]
num_bad = [(s, number_phonemes(s), t2p(s)) for s in num_samples if number_phonemes(s) != t2p(s)]
report['numbers'] = f'{len(num_samples) - len(num_bad)}/{len(num_samples)}'
for s, mine, ref in num_bad[:10]:
    print(f'NUM MISMATCH {s}: {ascii(mine)} vs {ascii(ref)}')

# ---- provider prototype: chunk -> raw, exactly what the C# HindiProvider will do ----
DANDAS = {'।', '॥'}
def native_chunk(chunk, oov):
    tokens = [t for t in chunk.split() if t.strip('।॥') or t in DANDAS or set(t) <= set('।॥')]
    spoken = [(t.strip('।॥') or t).replace('‌', '') for t in tokens]
    spoken = [t for t in spoken if t]
    rendered, said_something = [], False
    last_spoken = next((i for i in range(len(spoken) - 1, -1, -1) if spoken[i] not in DANDAS), -1)
    n_words = sum(1 for t in spoken if t not in DANDAS)
    wi = -1
    for token in spoken:
        if token in DANDAS:
            if not said_something: rendered.append(lexicon[token][0]); said_something = True
            continue
        wi += 1
        if re.fullmatch(r'[०-९0-9]+', token):
            rendered.append(number_phonemes(token)); said_something = True
        elif token in lexicon:
            alone, nonfinal, final = lexicon[token]
            rendered.append(alone if n_words == 1 else final if wi == n_words - 1 else nonfinal)
            said_something = True
        else:
            oov.add(token)
    return ' '.join(p for p in rendered if p)

FLAGS_RE = re.compile(r'\(.+?\)')
def postproc(raw):
    line = raw.replace('\r\n', '\n').replace(chr(0x361), '^').strip().replace('\n', ' ').replace('  ', ' ')
    line = FLAGS_RE.sub('', line)
    return ' '.join(w.strip() for w in line.split(' ') if w.strip())

# ---- held-out gate: fresh articles, chunk-level parity vs live espeak ----
sys.path.insert(0, r'C:\Users\lyrco\source\_repos\KokoroSharpBinaries\models_notpushed\misaki')
from phonemizer.punctuation import Punctuation
punctuator = Punctuation()

heldout = sentences_from(HELDOUT_TITLES, 1200)
gate = Counter()
oov_counter, mismatch_examples, heldout_rows = Counter(), [], []
for sent in heldout:
    chunks, _ = punctuator.preserve([sent])
    sent_ok, sent_oov = True, False
    for chunk in chunks:
        ref = t2p(chunk)
        oov = set()
        mine = native_chunk(chunk, oov)
        gate['chunks'] += 1
        if oov:
            gate['oov_chunks'] += 1
            oov_counter.update(oov)
            sent_oov = True
        elif postproc(mine) == postproc(ref):
            gate['match'] += 1
        else:
            gate['mismatch'] += 1
            sent_ok = False
            if len(mismatch_examples) < 25: mismatch_examples.append((chunk, postproc(mine), postproc(ref)))
    heldout_rows.append((sent, sent_ok, sent_oov))

covered = gate['chunks'] - gate['oov_chunks']
devanagari_oov = sum(1 for w in oov_counter if re.search(r'[ऀ-ॿ]', w))
report['heldout'] = {
    'sentences': len(heldout), 'chunks': gate['chunks'],
    'oov_chunk_pct': round(100 * gate['oov_chunks'] / gate['chunks'], 2), 'unique_oov_words': len(oov_counter),
    'oov_devanagari_vs_other': f'{devanagari_oov}/{len(oov_counter) - devanagari_oov}',
    'parity_on_covered_chunks': f"{gate['match']}/{covered} ({round(100 * gate['match'] / covered, 3)}%)",
    'clean_sentences_pct': round(100 * sum(1 for _, ok, oov in heldout_rows if ok and not oov) / len(heldout), 2),
}
for chunk, mine, ref in mismatch_examples[:10]:
    print(f'CHUNK MISMATCH {ascii(chunk)}\n  mine {ascii(mine)}\n  ref  {ascii(ref)}')
print('TOP OOV:', ' '.join(w for w, _ in oov_counter.most_common(15)))

# ---- emit lexicon + heldout fixture ----
def write_gz(path, text):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0) as f:
        f.write(text.encode('utf-8'))
    with open(path, 'wb') as f:
        f.write(buf.getvalue())
    return len(buf.getvalue())

lex_lines = ['# espeak-ng 1.52.0 hi dump, generated by generate_hi_native.py -- word\\talone[\\tnonfinal\\tfinal]']
lex_lines += [f'@@N\t{n}\t{ps}' for n, ps in NUM.items()]
lex_lines += [f'@@NH\t{d}\t{ps}' for d, ps in HUNDRED_FUSED.items()]
lex_lines += [f'@@NS\t{power}\t{ps}' for power, ps in SCALE_PS.items()]
lex_lines += [f'@@NT\t{n}\t{ps}' for n, ps in TAIL_AFTER_HUNDRED.items()]
lex_lines += [f'@@NF\t{n}\t{ps}' for n, ps in TAIL_FINAL.items()]
lex_lines += [f'@@NG\t{n}\t{ps}' for n, ps in GROUP.items()]
lex_lines += [f'@@ND\t{d}\t{ps}' for d, ps in DIGITWISE.items()]
for w, (alone, nonfinal, final) in sorted(lexicon.items()):
    assert '\t' not in alone + nonfinal + final and '\n' not in alone + nonfinal + final
    lex_lines.append(f'{w}\t{alone}' + (f'\t{nonfinal}\t{final}' if (nonfinal, final) != (alone, alone) else ''))
report['lexicon_gz_bytes'] = write_gz(os.path.join(DATA, 'hi_espeak_lexicon.tsv.gz'), '\n'.join(lex_lines) + '\n')

from misaki import espeak as misaki_espeak
g2p = misaki_espeak.EspeakG2P(language='hi')
fixture_lines = []
for sent, ok, oov in heldout_rows:
    ps, _ = g2p(sent)
    flag = 'oov' if oov else 'ok' if ok else 'diff'
    fixture_lines.append(f'{esc(sent)}\t{esc(ps)}\t{flag}')
report['heldout_fixture_gz_bytes'] = write_gz(os.path.join(FIXTURES, 'hi_native_heldout.tsv.gz'), '\n'.join(fixture_lines) + '\n')

print(json.dumps(report, indent=2, ensure_ascii=False))
