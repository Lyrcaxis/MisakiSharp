# Repacks unidic 3.1.0 (misaki[ja]'s dictionary) into the tables JapaneseTagger needs, then mass-verifies
# a pure-python replica of MeCab's lattice (prefix lookup, unknown-word rules, viterbi with its exact
# tie-breaking) against fugashi itself, so the C# port starts from a spec already proven bit-exact.
#
# Outputs (to REPACK, promoted into ../data once the packaging strategy is settled):
#   ja_lexicon.tsv    : surface<TAB>lc,rc,wcost,pron per token, token order preserved from sys.dic
#   ja_char.tsv       : 'c' category lines (name, invoke, group, length) + 'r' codepoint ranges (default, compat mask)
#   ja_unk.tsv        : category<TAB>lc,rc,wcost per unk token, unk.dic order
#   ja_matrix.bin     : verbatim matrix.bin (lsize, rsize, int16 grid) -- compression handled at pack time
import io, os, struct, sys
import numpy as np

DICDIR = r'C:\Users\lyrco\dev\libs\miniconda3\Lib\site-packages\unidic\dicdir'
MISAKI = r'C:\Users\lyrco\source\_repos\KokoroSharpBinaries\models_notpushed\misaki'
PARITY = r'C:\Users\lyrco\source\_repos\KokoroSharpBinaries\models_notpushed\parity'
REPACK = r'C:\Users\lyrco\source\_repos\KokoroSharpBinaries\models_notpushed\ja_repack'
MAX_GROUPING = 24
os.makedirs(REPACK, exist_ok=True)

# ---- sys.dic: header, darts double-array, token records, feature blob ----
def read_dic(path):
    data = open(path, 'rb').read()
    magic, version, dtype, lexsize, lsize, rsize, dsize, tsize, fsize, _dummy = struct.unpack_from('<10I', data, 0)
    assert magic ^ len(data) == 0xef718f77 and version == 102, (hex(magic), version)
    charset = data[40:72].split(b'\0')[0].decode()
    assert charset.lower() in ('utf-8', 'utf8'), charset
    darts = np.frombuffer(data, dtype=np.dtype([('base', '<i4'), ('check', '<u4')]), count=dsize // 8, offset=72)
    tokens = np.frombuffer(data, dtype=np.dtype([('lc', '<u2'), ('rc', '<u2'), ('posid', '<u2'), ('wcost', '<i2'), ('feature', '<u4'), ('compound', '<u4')]), count=tsize // 16, offset=72 + dsize)
    features = data[72 + dsize + tsize : 72 + dsize + tsize + fsize]
    return darts, tokens, features

def darts_keys(darts):
    """Enumerates every (utf8-key, value) in the double array via vectorized per-level BFS."""
    base, check = darts['base'].astype(np.int64), darts['check'].astype(np.int64)
    keys, frontier = [], [(int(base[0]), b'')]
    while frontier:
        bases = np.array([b for b, _ in frontier], dtype=np.int64)
        terminal = (check[bases] == bases) & (base[bases] < 0)
        for i in np.flatnonzero(terminal):
            keys.append((frontier[i][1], int(-base[bases[i]] - 1)))
        nxt = []
        probes = bases[:, None] + np.arange(1, 257)[None, :]
        hits = check[probes] == bases[:, None]
        for i, j in zip(*np.nonzero(hits)):
            nxt.append((int(base[probes[i, j]]), frontier[i][1] + bytes([j])))
        frontier = nxt
    return keys

def parse_feature(blob, offset):
    """CSV fields of one feature string (quoted fields may contain commas)."""
    raw = blob[offset:blob.index(b'\0', offset)].decode()
    fields, cur, quoted = [], '', False
    for ch in raw:
        if ch == '"': quoted = not quoted
        elif ch == ',' and not quoted: fields.append(cur); cur = ''
        else: cur += ch
    fields.append(cur)
    return fields

darts, tokens, features = read_dic(os.path.join(DICDIR, 'sys.dic'))
lexicon = {}  # surface -> [(lc, rc, wcost, pron)]
for key, value in darts_keys(darts):
    surface = key.decode()
    entries = []
    for tok in tokens[value >> 8 : (value >> 8) + (value & 0xff)]:
        fields = parse_feature(features, int(tok['feature']))
        assert len(fields) == 29 and fields[9], (surface, fields)
        entries.append((int(tok['lc']), int(tok['rc']), int(tok['wcost']), fields[9]))
    lexicon[surface] = entries
print(f'lexicon: {len(lexicon)} surfaces, {sum(len(v) for v in lexicon.values())} tokens')

# ---- char.def: categories + codepoint ranges (cross-checked against compiled char.bin) ----
categories, ranges = {}, []  # name -> (id, invoke, group, length); (lo, hi, default_id, compat_mask)
for line in io.open(os.path.join(DICDIR, 'char.def'), encoding='utf-8'):
    line = line.split('#')[0].strip()
    if not line: continue
    parts = line.split()
    if not parts[0].startswith('0x'):
        categories[parts[0]] = (len(categories), int(parts[1]), int(parts[2]), int(parts[3]))
    else:
        lo, hi = (parts[0].split('..') + [parts[0]])[:2]
        mask = sum(1 << categories[name][0] for name in parts[1:])
        ranges.append((int(lo, 16), int(hi, 16), categories[parts[1]][0], mask))

# fugashi's mecab maps codepoints 0..0xFFFE (0xffff entries); astral chars index via truncated ucs2.
charmap = np.zeros(0xffff, dtype=np.uint32)  # CharInfo bitfield: type:18 | default:8 | length:4 | group:1 | invoke:1
default_id, default_invoke, default_group, default_length = categories['DEFAULT'][0], *categories['DEFAULT'][1:]
charmap[:] = (1 << default_id) | (default_id << 18) | (default_length << 26) | (default_group << 30) | (default_invoke << 31)
for lo, hi, default, mask in ranges:
    if lo > 0xfffe: continue
    _, invoke, group, length = [v for v in categories.values() if v[0] == default][0]
    charmap[lo:min(hi, 0xfffe) + 1] = mask | (default << 18) | (length << 26) | (group << 30) | (invoke << 31)

charbin = open(os.path.join(DICDIR, 'char.bin'), 'rb').read()
csize = struct.unpack_from('<I', charbin)[0]
binmap = np.frombuffer(charbin, dtype='<u4', count=0xffff, offset=4 + 32 * csize)
assert csize == len(categories) and np.array_equal(charmap, binmap), 'char.def does not reproduce char.bin'
print(f'char: {len(categories)} categories, {len(ranges)} ranges, char.bin cross-check OK')

# ---- unk.dic: per-category unknown tokens (cross-checked against unk.def text) ----
unk_darts, unk_tokens, unk_features = read_dic(os.path.join(DICDIR, 'unk.dic'))
unk = {}  # category -> [(lc, rc, wcost)]
for key, value in darts_keys(unk_darts):
    unk[key.decode()] = [(int(t['lc']), int(t['rc']), int(t['wcost'])) for t in unk_tokens[value >> 8 : (value >> 8) + (value & 0xff)]]
assert set(unk) == set(categories), (set(unk) ^ set(categories))
print(f'unk: {sum(len(v) for v in unk.values())} tokens across {len(unk)} categories')

# ---- matrix.bin ----
matbytes = open(os.path.join(DICDIR, 'matrix.bin'), 'rb').read()
msize_l, msize_r = struct.unpack_from('<2H', matbytes)
matrix = np.frombuffer(matbytes, dtype='<i2', count=msize_l * msize_r, offset=4).reshape(msize_r, msize_l).T  # [rcAttr, lcAttr]
print(f'matrix: {msize_l} x {msize_r}')

# ---- python replica of MeCab's lattice (tokenizer.cpp + viterbi.cpp semantics) ----
CINFO = charmap  # per-codepoint bitfield; mecab's ucs2 decoder yields 0 for astral chars, so they take codepoint 0's category
def char_info(ch):
    return int(CINFO[ord(ch)]) if ord(ch) < 0xffff else int(CINFO[0])
def cat_default(ci): return (ci >> 18) & 0xff
def cat_length(ci): return (ci >> 26) & 0xf
def cat_group(ci): return (ci >> 30) & 0x1
def cat_invoke(ci): return (ci >> 31) & 0x1
def is_kind_of(ci, other): return (ci & other & 0x3ffff) != 0

MAXLEN = max(len(s) for s in lexicon)
SPACE_INFO = char_info(' ')
cat_names = {v[0]: k for k, v in categories.items()}

def replica_parse(text):
    """Returns [(surface, pron_or_None, char_type, is_unk)] for the best path, MeCab-identical."""
    n = len(text)
    ends = [[] for _ in range(n + 1)]  # (cost, rc, node) prepend-ordered like end_node_list_
    ends[0].append((0, 0, None))
    results = [None] * (n + 1)
    for pos in range(n + 1):
        if not ends[pos] or pos == n: continue
        # leading SPACE-category chars attach to the node's span but not its surface
        begin2 = pos
        while begin2 < n and is_kind_of(SPACE_INFO, char_info(text[begin2])): begin2 += 1
        nodes = []  # built in mecab's linked-list order: prepend as created
        found = False
        for length in range(1, min(MAXLEN, n - begin2) + 1):
            entries = lexicon.get(text[begin2:begin2 + length])
            if entries is None: continue
            found = True
            for lc, rc, wcost, pron in entries:
                nodes.insert(0, (begin2 + length, lc, rc, wcost, text[begin2:begin2 + length], pron, False))
        if begin2 < n:
            ci = char_info(text[begin2])
            if cat_invoke(ci) or not found:
                cat = cat_names[cat_default(ci)]
                run_end, extra = begin2 + 1, 0  # mecab's clen counts chars beyond the first, so groups span up to MAX_GROUPING+1
                while run_end < n and is_kind_of(ci, char_info(text[run_end])): run_end, extra = run_end + 1, extra + 1
                group_end = 0
                if cat_group(ci):
                    if extra <= MAX_GROUPING:
                        for lc, rc, wcost in unk[cat]:
                            nodes.insert(0, (run_end, lc, rc, wcost, text[begin2:run_end], None, True))
                    group_end = run_end
                for i in range(1, cat_length(ci) + 1):
                    if begin2 + i > run_end or begin2 + i == group_end: break
                    for lc, rc, wcost in unk[cat]:
                        nodes.insert(0, (begin2 + i, lc, rc, wcost, text[begin2:begin2 + i], None, True))
        if not nodes: continue
        lefts_cost = np.array([e[0] for e in ends[pos]], dtype=np.int64)
        lefts_rc = np.array([e[1] for e in ends[pos]], dtype=np.int64)
        for node in nodes:
            total = lefts_cost + matrix[lefts_rc, node[1]] + node[3]
            best = int(np.argmin(total))
            ends[node[0]].insert(0, (int(total[best]), node[2], (node, ends[pos][best])))
    eos_pos = next(pos for pos in range(n, -1, -1) if ends[pos])  # eos absorbs trailing spaces, like mecab's backward scan
    lefts_cost = np.array([e[0] for e in ends[eos_pos]], dtype=np.int64)
    lefts_rc = np.array([e[1] for e in ends[eos_pos]], dtype=np.int64)
    best = int(np.argmin(lefts_cost + matrix[lefts_rc, 0]))
    path, entry = [], ends[eos_pos][best][2]
    while entry is not None:
        node, prev_end = entry
        path.append((node[4], node[5], cat_default(char_info(node[4][0])), node[6]))
        entry = prev_end[2]
    return path[::-1]

# ---- mass verify vs fugashi over the wiki corpus (raw + misaki-normalized) ----
sys.path.insert(0, MISAKI)
from misaki.cutlet import Cutlet
from fugashi import Tagger
tagger, cutlet = Tagger(), Cutlet()

def fugashi_parse(text):
    return [(w.surface, w.feature.pron, w.char_type, bool(w.is_unk)) for w in tagger(text)]

import re
sentences = []
for fname in sorted(os.listdir(PARITY)):
    if not fname.startswith('wiki_ja_'): continue
    text = ' '.join(io.open(os.path.join(PARITY, fname), encoding='utf-8').read().split('\n'))
    sentences += [s.strip() for s in re.split(r'(?<=[。！？.!?…])', text) if 10 <= len(s.strip()) <= 180]
sentences = sorted(set(sentences))
sentences += ['こんにちは ', 'テスト  テスト', '   ', '𩸽の刺身と😀絵文字', 'ヷヸヹヺ', 'ｱｲｳｴｵ', 'A B C', '「 」', '一二三 ']
print(f'verify corpus: {len(sentences)} sentences x2 (raw + normalized)')

mismatches = 0
for si, sent in enumerate(sentences):
    for text in (sent, cutlet._normalize_text(sent)):
        expected, got = fugashi_parse(text), replica_parse(text)
        if expected != got:
            mismatches += 1
            if mismatches <= 5:
                print(f'MISMATCH {ascii(text[:60])}')
                for a, b in zip(expected, got):
                    if a != b: print('  fugashi', a, ' replica', b)
    if (si + 1) % 500 == 0: print(f'  {si + 1}/{len(sentences)}, {mismatches} mismatches')
print(f'replica parity: {2 * len(sentences) - mismatches}/{2 * len(sentences)}')

# ---- dump: embeddable tables into ../data, the 481MB matrix stays a file (MisakiSharp.MatrixPath) ----
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Tests', 'fixtures')
def esc(s):
    return s.replace('\\', '\\\\').replace('\t', '\\t').replace('\n', '\\n').replace('\r', '\\r')

import gzip
def write_gz(path, lines):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0) as f:
        f.write(('\n'.join(lines) + '\n').encode('utf-8'))
    open(path, 'wb').write(buf.getvalue())
    print(path, len(buf.getvalue()))

lex_lines = [esc(surface) + '\t' + '\t'.join(f'{lc},{rc},{wcost},{esc(pron)}' for lc, rc, wcost, pron in lexicon[surface]) for surface in sorted(lexicon)]
write_gz(os.path.join(DATA, 'ja_lexicon.tsv.gz'), lex_lines)
charunk_lines = [f'c\t{name}\t{cid}\t{invoke}\t{group}\t{length}' for name, (cid, invoke, group, length) in categories.items()]
charunk_lines += [f'r\t{lo}\t{hi}\t{default}\t{mask}' for lo, hi, default, mask in ranges]
charunk_lines += [f'u\t{cat}\t' + '\t'.join(f'{lc},{rc},{wcost}' for lc, rc, wcost in unk[cat]) for cat in sorted(unk)]
write_gz(os.path.join(DATA, 'ja_char_unk.tsv.gz'), charunk_lines)
import shutil
shutil.copyfile(os.path.join(DICDIR, 'matrix.bin'), os.path.join(REPACK, 'ja_matrix.bin'))
shutil.copyfile(os.path.join(DICDIR, 'matrix.bin'), os.path.join(FIXTURES, 'ja_matrix.bin'))  # gitignored, tests point here
print('matrix copied to', REPACK, 'and', FIXTURES)
