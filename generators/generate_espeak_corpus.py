# Harvest a large real-text vocabulary per language from a Wikipedia dump prefix.
# Multistream bz2 decodes fine from a truncated prefix, so a ranged GET is enough.
# Writes <out>/<code>_corpus_words.txt, frequency-ordered -- picked up by the espeak dump's build_vocab.
import bz2, io, os, re, sys, time, urllib.request
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

OUT = r'C:\Users\lyrco\source\_repos\KokoroSharpBinaries\models_notpushed\espeak_dump'
WIKI = {'es': 'eswiki', 'fr': 'frwiki', 'it': 'itwiki', 'pt': 'ptwiki', 'hi': 'hiwiki'}
BYTES = int(os.environ.get('MB', 120)) * 1_000_000
MIN_COUNT = 2

MARKUP = re.compile(r'(?s)<ref.*?</ref>|<[^>]+>|\{\{.*?\}\}|\{\|.*?\|\}|\[\[[^]|]*\||https?://\S+|[\[\]{}|=*#:;\']')
TOKEN = re.compile(r"[^\W\d_](?:[\w'\u2019.-]*[^\W\d_])?", re.UNICODE)


def fetch(code):
    name = WIKI[code]
    cache = os.path.join(OUT, f'{name}_prefix.bz2')
    # A cache left short by an interrupted run must be refetched, not silently decoded to nothing.
    if not os.path.exists(cache) or os.path.getsize(cache) < BYTES:
        url = f'https://dumps.wikimedia.org/{name}/latest/{name}-latest-pages-articles-multistream.xml.bz2'
        req = urllib.request.Request(url, headers={
            'User-Agent': 'MisakiSharpFixtures/1.0 (https://github.com/Lyrcaxis/MisakiSharp; lyrcaxis@gmail.com)',
            'Range': f'bytes=0-{BYTES - 1}'})
        start = time.time()
        with urllib.request.urlopen(req, timeout=600) as r, open(cache, 'wb') as f:
            f.write(r.read())
        print(f'[{code}] downloaded {os.path.getsize(cache) / 1e6:.0f} MB in {time.time() - start:.0f}s', flush=True)
    return cache


def words_of(code):
    """A multistream dump is many concatenated bz2 streams -- each end-of-stream needs a fresh decompressor."""
    path = fetch(code)
    decomp = bz2.BZ2Decompressor()
    counts, pending, streams = Counter(), b'', 0
    with open(path, 'rb') as f:
        while chunk := f.read(4 << 20):
            data = chunk
            while data:
                try:
                    pending += decomp.decompress(data)
                except (EOFError, OSError):
                    data = b''
                    break
                if not decomp.eof: break
                data, decomp, streams = decomp.unused_data, bz2.BZ2Decompressor(), streams + 1
            if len(pending) > (8 << 20):
                counts.update(TOKEN.findall(MARKUP.sub(' ', pending.decode('utf-8', 'ignore'))))
                pending = b''
    if pending:
        counts.update(TOKEN.findall(MARKUP.sub(' ', pending.decode('utf-8', 'ignore'))))
    print(f'[{code}] decoded {streams} streams', flush=True)
    return counts


for code in sys.argv[1:] or list(WIKI):
    start = time.time()
    counts = words_of(code)
    kept = [w for w, n in counts.most_common() if n >= MIN_COUNT and len(w) <= 40]
    with io.open(os.path.join(OUT, f'{code}_corpus_words.txt'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(kept) + '\n')
    print(f'[{code}] {time.time() - start:.0f}s  unique={len(counts)}  kept(>={MIN_COUNT})={len(kept)}', flush=True)
