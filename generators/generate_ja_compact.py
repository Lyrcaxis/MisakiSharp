# Compresses unidic 3.1.0's 481MB connection matrix into the ~8MB table JapaneseTagger embeds.
#
# The matrix is incompressible losslessly (12.37 bits/entry of true entropy), but it is very low-rank
# AND only ~1% of its 240M entries are ever consulted, so: int8 low-rank base + exact values cached
# for the pairs real text actually hits. Reconstruction is pure integer arithmetic (power-of-two
# per-component scales), so C# reproduces this bit-for-bit on any platform.
#
# Pair frequencies come from ja_record_norm.py, which must record over NORMALIZED text -- JAG2P applies
# NFKC+mojimoji before tagging, so recording raw sentences misses the pairs production actually consults.
#
# Output: ../data/ja_matrix_compact.bin.gz
#   u16 leftSize, u16 rightSize, u16 rank, u16 shift, i16 grand
#   i16 rowMean[rightSize], i16 colMean[leftSize], u8 shiftA[rank]
#   i8  A[rightSize*rank], i8 B[leftSize*rank]
#   i32 rowOffsets[leftSize+1], u16 cacheCols[n], i16 cacheVals[n]     (CSR over left rcAttr)
import gzip, os, struct
import numpy as np
from sklearn.utils.extmath import randomized_svd

DICDIR = r'C:\Users\lyrco\dev\libs\miniconda3\Lib\site-packages\unidic\dicdir'
PAIRS = r'C:\Users\lyrco\source\_repos\KokoroSharpBinaries\models_notpushed\ja_repack'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'ja_matrix_compact.bin.gz')
RANK, MIN_COUNT = 128, 1

raw = np.fromfile(os.path.join(DICDIR, 'matrix.bin'), dtype=np.uint8)
leftSize, rightSize = struct.unpack('<HH', raw[:4].tobytes())
exact = raw[4:].view(np.int16).reshape(rightSize, leftSize).T          # [leftRc, rightLc]
print(f'matrix {leftSize} x {rightSize} ({exact.nbytes/1e6:.0f} MB)')

m = exact.T.astype(np.float32)                                          # [rightLc, leftRc]
rowMean = np.rint(m.mean(axis=1)).astype(np.int32)                      # per rightLc
colMean = np.rint(m.mean(axis=0)).astype(np.int32)                      # per leftRc
grand = int(round(float(m.mean())))
U, S, Vt = randomized_svd(m - m.mean(axis=1, keepdims=True) - m.mean(axis=0, keepdims=True) + m.mean(),
                          n_components=RANK, n_iter=4, random_state=0)

A = (U * S).astype(np.float64)                                          # (rightSize, RANK)
B = Vt.T.astype(np.float64)                                             # (leftSize, RANK)
expA = np.floor(np.log2(127.0 / np.abs(A).max(axis=0))).astype(np.int64)
expB = int(np.floor(np.log2(127.0 / np.abs(B).max())))
Aq = np.clip(np.rint(A * (2.0 ** expA)), -127, 127).astype(np.int8)
Bq = np.clip(np.rint(B * (2.0 ** expB)), -127, 127).astype(np.int8)
shift = int(expA.max()) + expB
shiftA = (int(expA.max()) - expA).astype(np.uint8)                      # per-component left shift, >= 0

acc = (Aq.astype(np.float64) * (2.0 ** shiftA)) @ Bq.astype(np.float64).T   # exact in float64
base = np.floor((acc + (1 << (shift - 1))) / float(1 << shift)).T + rowMean[None, :] + colMean[:, None] - grand
base = np.clip(base, -32768, 32767).astype(np.int16)
del acc

keys = np.load(os.path.join(PAIRS, 'norm_keys_450000.npy'))
counts = np.load(os.path.join(PAIRS, 'norm_counts_450000.npy'))
keys = np.sort(keys[counts >= MIN_COUNT])
rows, cols = (keys // rightSize).astype(np.int64), (keys % rightSize).astype(np.uint16)
vals = exact[rows, cols]
wrong = int((base[rows, cols] != vals).sum())
rowOffsets = np.zeros(leftSize + 1, dtype=np.int32)
np.cumsum(np.bincount(rows, minlength=leftSize), out=rowOffsets[1:])
print(f'cache {len(keys):,} pairs ({len(keys)/exact.size*100:.2f}% of matrix), {wrong:,} of them the base got wrong')

blob = (struct.pack('<4Hh', leftSize, rightSize, RANK, shift, grand)
        + rowMean.astype(np.int16).tobytes() + colMean.astype(np.int16).tobytes() + shiftA.tobytes()
        + Aq.tobytes() + Bq.tobytes()
        + rowOffsets.tobytes() + cols.tobytes() + vals.astype(np.int16).tobytes())
os.makedirs(os.path.dirname(OUT), exist_ok=True)
packed = gzip.compress(blob, 9)
open(OUT, 'wb').write(packed)
print(f'{OUT} -> {len(packed)/1e6:.2f} MB gz ({len(blob)/1e6:.1f} MB resident)')
