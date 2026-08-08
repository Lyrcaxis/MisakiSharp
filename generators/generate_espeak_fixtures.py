# Oracle-generates parity fixtures for the EspeakG2P/EspeakFallback port (es/fr/hi/it/pt + English OOV fallback).
#
# Runs misaki 0.9.4's espeak module (phonemizer + espeak-ng 1.52.0 via espeakng_loader) over held-out Wikipedia
# sentences, recording BOTH the final phonemes and every raw per-chunk espeak response the backend saw.
# The C# tests replay the recorded chunks through the ported pipeline and demand byte parity on the final output,
# so the port is verified end-to-end without spawning espeak. A separate section mass-checks that the espeak-ng
# CLI (`--ipa --tie=^`, the invocation MisakiSharp's ProcessProvider uses) reproduces the library's raw output.
#
# Fixture formats (UTF-8, '\n' newlines, inside .gz; '\\', '\t', '\n', '\r' escaped as literal backslash pairs):
#   espeak_{lang}_heldout.tsv.gz : sentence<TAB>expected_phonemes
#   espeak_{lang}_chunks.tsv.gz  : chunk<TAB>raw_espeak_response      (also for en_us/en_gb, feeding the fallback)
#   espeak_en_fallback.tsv.gz    : word<TAB>us_phonemes<TAB>gb_phonemes   ('∅' encodes misaki's None)
import gzip, io, json, os, random, re, subprocess, sys, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

MISAKI = r'C:\Users\lyrco\source\_repos\KokoroSharpBinaries\models_notpushed\misaki'
PARITY = r'C:\Users\lyrco\source\_repos\KokoroSharpBinaries\models_notpushed\parity'
OUT = r'C:\Users\lyrco\source\repos\MisakiSharp\Tests\fixtures'
ESPEAK_EXE = r'C:\Users\lyrco\source\repos\KokoroSharp\bin\Debug\net8.0\espeak\espeak-ng.exe'
ESPEAK_DATA = r'C:\Users\lyrco\source\repos\KokoroSharp\bin\Debug\net8.0\espeak\espeak-ng-data'
sys.path.insert(0, MISAKI)
from misaki import espeak

LANGS = {'es': 'es', 'fr': 'fr-fr', 'hi': 'hi', 'it': 'it', 'pt': 'pt-br'}
WIKI_TITLES = {
    'es': ['España', 'Miguel de Cervantes', 'Historia de España', 'Matemáticas', 'Segunda Guerra Mundial', 'Sol', 'Música', 'Ciencia', 'Madrid', 'Física', 'Universo', 'Internet', 'Agua', 'Luna', 'Literatura', 'Química'],
    'fr': ['France', 'Paris', 'Histoire de France', 'Mathématiques', 'Seconde Guerre mondiale', 'Soleil', 'Musique', 'Science', 'Physique', 'Univers', 'Internet', 'Eau', 'Lune', 'Napoléon Ier', 'Littérature', 'Chimie'],
    'hi': ['भारत', 'हिन्दी', 'दिल्ली', 'महात्मा गांधी', 'विज्ञान', 'गणित', 'सूर्य', 'चन्द्रमा', 'संगीत', 'जल', 'इतिहास', 'भौतिक शास्त्र', 'अन्तरजाल', 'पृथ्वी', 'ताजमहल', 'रसायन विज्ञान'],
    'it': ['Italia', 'Roma', "Storia d'Italia", 'Matematica', 'Seconda guerra mondiale', 'Sole', 'Musica', 'Scienza', 'Fisica', 'Universo', 'Internet', 'Acqua', 'Luna', 'Dante Alighieri', 'Letteratura', 'Chimica'],
    'pt': ['Brasil', 'Portugal', 'São Paulo', 'Matemática', 'Segunda Guerra Mundial', 'Sol', 'Música', 'Ciência', 'Física', 'Universo', 'Internet', 'Água', 'Lua', 'História do Brasil', 'Literatura', 'Química'],
    'en': ['United States', 'World War II', 'Mathematics', 'Physics', 'Universe', 'Internet', 'Water', 'Moon', 'Literature', 'Chemistry', 'William Shakespeare', 'London'],
}
rng = random.Random(42)
report = {}

# ---- chunk recording: every raw text_to_phonemes response, keyed by the active voice ----
import phonemizer.backend.espeak.wrapper as wrapper_mod
RECORDED, ACTIVE_VOICE = {}, ['?']
_orig_t2p = wrapper_mod.EspeakWrapper.text_to_phonemes
def _recording_t2p(self, text, tie=False):
    raw = _orig_t2p(self, text, tie)
    RECORDED.setdefault(ACTIVE_VOICE[0], {}).setdefault(text, raw)
    return raw
wrapper_mod.EspeakWrapper.text_to_phonemes = _recording_t2p

# ---- corpus: full-article plaintext from each wiki, cached locally ----
def fetch_wiki(code, title):
    cache = os.path.join(PARITY, f'wiki_{code}_{re.sub(r"[^\w]+", "_", title)}.txt')
    if os.path.exists(cache):
        return io.open(cache, encoding='utf-8').read()
    url = (f'https://{code}.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&format=json'
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
    time.sleep(1)
    assert text, f'empty extract for {code}:{title}'
    io.open(cache, 'w', encoding='utf-8', newline='\n').write(text)
    return text

def corpus_sentences(code, count=1600):
    lines = []
    for title in WIKI_TITLES[code]:
        text = fetch_wiki(code, title)
        lines += [l for l in text.split('\n') if l.strip() and not re.fullmatch(r'=+ .+? =+', l.strip())]
    sentences, seen = [], set()
    for sent in re.split(r'(?<=[.!?…।])\s+', ' '.join(lines)):
        sent = re.sub(r'\s+', ' ', sent).strip()
        if 20 <= len(sent) <= 180 and re.search(r'\w', sent) and sent not in seen:
            seen.add(sent)
            sentences.append(sent)
    assert len(sentences) >= count, f'{code}: only {len(sentences)} sentences'
    return rng.sample(sentences, count)

def esc(s):
    return s.replace('\\', '\\\\').replace('\t', '\\t').replace('\n', '\\n').replace('\r', '\\r')

def write_gz(name, lines):
    os.makedirs(OUT, exist_ok=True)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0) as f:
        f.write(('\n'.join(lines) + '\n').encode('utf-8'))
    with open(os.path.join(OUT, name), 'wb') as f:
        f.write(buf.getvalue())
    return len(buf.getvalue())

# ---- oracle: EspeakG2P per language (sequential -- espeak voice state is process-global) ----
for code, voice in LANGS.items():
    ACTIVE_VOICE[0] = voice
    g2p = espeak.EspeakG2P(language=voice)
    pairs = []
    for sent in corpus_sentences(code):
        ps, _ = g2p(sent)
        assert '\n' not in ps and '\t' not in sent, repr(sent)
        pairs.append(f'{esc(sent)}\t{esc(ps)}')
    write_gz(f'espeak_{code}_heldout.tsv.gz', pairs)
    chunk_bytes = write_gz(f'espeak_{code}_chunks.tsv.gz', [f'{esc(c)}\t{esc(r)}' for c, r in sorted(RECORDED[voice].items())])
    report[code] = dict(sentences=len(pairs), chunks=len(RECORDED[voice]), chunk_gz_bytes=chunk_bytes)

# ---- oracle: EspeakFallback words (wiki-en vocabulary + seeded fuzz + quirky formats) ----
tokens = sorted({t for title in WIKI_TITLES['en'] for t in re.findall(r"[A-Za-z][A-Za-z'.\-]*[A-Za-z.]|[A-Za-z]", fetch_wiki('en', title))})
words = rng.sample(tokens, 2500)
words += [''.join(rng.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(rng.randint(3, 12))) for _ in range(400)]
words += ["can't", "won't", 'U.S.A.', 'iPhone7', 'e.g.', 'anti-hero', 'naïve', 'Zürich', 'café', 'MREs', 'DDoS', 'qwrtzx', 'ooo', 'xylo']
class Tok: pass
fallback_lines = []
for british, key in ((False, 'en-us'), (True, 'en-gb')):
    ACTIVE_VOICE[0] = key
    fb = espeak.EspeakFallback(british=british)
    results = []
    for w in words:
        tok = Tok(); tok.text = w
        ps, _ = fb(tok)
        assert ps != '∅', w
        results.append('∅' if ps is None else ps)
    fallback_lines.append(results)
write_gz('espeak_en_fallback.tsv.gz', [f'{esc(w)}\t{esc(us)}\t{esc(gb)}' for w, us, gb in zip(words, *fallback_lines)])
for key, name in (('en-us', 'en_us'), ('en-gb', 'en_gb')):
    write_gz(f'espeak_{name}_chunks.tsv.gz', [f'{esc(c)}\t{esc(r)}' for c, r in sorted(RECORDED[key].items())])
report['en_fallback'] = dict(words=len(words), us_chunks=len(RECORDED['en-us']), gb_chunks=len(RECORDED['en-gb']))

# ---- CLI equivalence: the ProcessProvider invocation must reproduce the library's raw responses ----
def postprocessed(raw):
    line = raw.replace('\r\n', '\n').replace(chr(0x361), '^').strip().replace('\n', ' ').replace('  ', ' ')
    return ' '.join(w.strip() for w in line.split(' '))

def cli_matches(voice, chunk, raw):
    env = dict(os.environ, ESPEAK_DATA_PATH=ESPEAK_DATA)
    out = subprocess.run([ESPEAK_EXE, '--ipa', '--tie=^', '-q', '-b', '1', '-v', voice, '--stdin'],
                         input=(chunk + '\n').encode('utf-8'), capture_output=True, env=env).stdout.decode('utf-8')
    return postprocessed(out) == postprocessed(raw), chunk, out

for voice, recorded in RECORDED.items():
    sample = rng.sample(sorted(recorded.items()), min(600, len(recorded)))
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda kv: cli_matches(voice, *kv), sample))
    bad = [(c, o) for ok, c, o in results if not ok]
    report.setdefault('cli_vs_lib', {})[voice] = f'{len(sample) - len(bad)}/{len(sample)}'
    for c, o in bad[:3]:
        print(f'CLI MISMATCH [{voice}] {ascii(c)} -> {ascii(o)} (lib: {ascii(recorded[c])})')

print(json.dumps(report, indent=2, ensure_ascii=False))
