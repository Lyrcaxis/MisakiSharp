# Adds the symbol and zero-padded-number tokens to an existing dump, reusing the reference contexts the
# full run already discovered. Two thirds of what still falls out of vocabulary in es/fr/it/pt is '%', '000',
# '°C', '/' and friends -- and since espeak phonemizes '8%' exactly as '8' followed by '%', giving the pieces
# their own entries lets the boundary model assemble the mixed tokens for free.
#
# Incremental on purpose: pass A discovers context classes from the whole vocabulary, so re-running it for a
# few thousand extra tokens would reshuffle the class ids every existing row is written against.
import gzip, io, os, sys, time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_espeak_native as G

sys.stdout.reconfigure(encoding='utf-8')

SYMBOLS = list('%°/&=+×–—-*~³²½§€$£¥ªº‰·±÷≈≠≤≥⊙@#^|\\<>') + ['°C', '°F', '...', '…']
DEVANAGARI = str.maketrans('0123456789', '०१२३४५६७८९')


def extra_tokens(code):
    """Zero-padded runs ('000' is a real token in '1 000 000'), plus the symbols, minus whatever is present."""
    out = [str(n).zfill(width) for width in (2, 3, 4) for n in range(10 ** width) if len(str(n)) < width]
    if code == 'hi': out += [t.translate(DEVANAGARI) for t in out]
    return out + SYMBOLS


def load_rows(code):
    rows, header = {}, None
    for line in gzip.open(os.path.join(G.OUT, f'{code}_words6.tsv.gz'), 'rt', encoding='utf-8'):
        line = line.rstrip('\n')
        if line.startswith('#'):
            header = line
        elif line:
            rows[line.split('\t')[0]] = line
    return header, rows


def load_contexts(code):
    reps, ctx_of = [], {}
    with io.open(os.path.join(G.OUT, f'{code}_ctx.tsv'), encoding='utf-8') as f:
        for line in f:
            index, rep, signature, _ = line.rstrip('\n').split('\t')
            reps.append(rep)
            ctx_of[tuple(signature.split(' | '))] = int(index)
    return reps, ctx_of


def main():
    for code in sys.argv[1:] or list(G.LANGS):
        voice, _, base, diags, nasal, vowel = G.LANGS[code]
        header, rows = load_rows(code)
        reps, ctx_of = load_contexts(code)
        tokens = [t for t in extra_tokens(code) if t not in rows]
        start = time.time()
        with ProcessPoolExecutor(os.cpu_count(), initializer=G._init,
                                 initargs=(voice, base, diags, nasal, vowel)) as pool:
            sigs = dict(pool.map(G.pass_a, tokens, chunksize=64))
        with ProcessPoolExecutor(os.cpu_count(), initializer=G._init,
                                 initargs=(voice, base, diags, nasal, vowel, reps)) as pool:
            results = list(pool.map(G.pass_b, tokens, chunksize=64))
        added = 0
        for word, alone, init, on, ol, tails, _ in results:
            ctx = ctx_of.get(sigs.get(word), -1)
            rows[word] = '\t'.join([word, alone, '=' if init == alone else init, on, ol,
                                    str(ctx) if ctx >= 0 else '', *tails]).rstrip('\t')
            added += 1
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0) as f:
            f.write(('\n'.join([header] + list(rows.values())) + '\n').encode('utf-8'))
        with open(os.path.join(G.OUT, f'{code}_words6.tsv.gz'), 'wb') as f:
            f.write(buf.getvalue())
        print(f'[{code}] +{added} tokens -> {len(rows)} rows  {time.time() - start:.0f}s  '
              f'gz={len(buf.getvalue()) / 1e6:.2f} MB', flush=True)


if __name__ == '__main__':
    main()
