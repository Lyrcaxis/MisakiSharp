# -*- coding: utf-8 -*-
"""Regenerates KokoroSharp's zh data tables for the misaki v1.1 (0.9.4) frontend.

Replicates misaki/zh_frontend.py `_init_pypinyin` exactly before dumping anything:
    large_pinyin.load()                          (pypinyin_dict phrase overrides)
    load_phrases_dict(<misaki custom phrases>)   (later loads override earlier ones)
    load_single_dict({ord('地'): 'de,di4'})
All emitted pinyin comes from REAL `lazy_pinyin(key, neutral_tone_with_five=True, style=...)`
calls (styles TONE3 / INITIALS / FINALS_TONE3, strict=True) -- never hand-derived.

============================== zh_g2p_v2.tsv.gz ==============================
UTF-8 TSV, gzip (mtime=0, deterministic). Line kinds, in file order:

  #<syl>\t<initial>,<final_tone3>
      One line per distinct TONE3 syllable seen across all entries below.
      <initial>/<final_tone3> are the RAW strict pypinyin outputs (ii/iii
      discrimination and the 嗯->n2 patch happen in the C# frontend, not here).
      Either side may be empty (e.g. '#n2\t,' for 嗯 whose strict initial AND
      final are both empty).

  <hanzi>\t<syl syl ...>
      Single char (len 1) or phrase (len >= 2) -> space-joined TONE3 syllables.
      Filtered to chars 一-鿿 (U+4E00..U+9FFF); phrases outside that range are
      unreachable through jieba.posseg words (losslessly dropped).
      Lookup contract for the C# frontend (mirrors pypinyin's mmseg with
      no_non_phrases=True): scan left to right, take the LONGEST phrase entry
      starting at the current position, else consume one char via its char
      entry. An in-range char with no entry yields initial='' AND final=''
      (pypinyin style-converts the raw han char, producing empties); non-han
      runs (ascii etc.) pass through RAW as ONE item per run for both styles,
      matching pypinyin errors='default'.

  @@\t<total>
      jieba.dt.total (sum of ALL dict.txt line freqs, duplicate lines included).

  @<word>\t<freq> <POS>
      One line per jieba dict.txt word (jieba.dt.FREQ > 0), POS from dict.txt
      (duplicate words: last line wins, same as jieba). The DAG prefix set
      (freq==0 entries of jieba.dt.FREQ) is derivable: every proper prefix of
      every @word, freq 0 unless itself an @word.

============================ zh_posseg_hmm.tsv.gz ============================
jieba.posseg HMM tables for OOV tagging (char_state_tab, prob_start, prob_trans,
prob_emit), exported for posseg.viterbi. Line kinds, in file order:

  S\t<idx>\t<B|M|E|S>,<POS>     state table; idx = 0..N-1, states sorted.
  P\t<idx>\t<logp>              prob_start; ABSENT idx => MIN_FLOAT (-3.14e100).
  T\t<from>\t<to>:<logp> ...    prob_trans row (space-joined, sorted by <to>);
                                absent (from,to) pair => -inf; absent row => {}.
  E\t<idx>\t<char>:<logp> ...   prob_emit row (sorted by char); absent char => MIN_FLOAT.
  C\t<char>\t<idx> <idx> ...    char_state_tab; chars not listed => all states.

Floats are printed with repr() (shortest exact round-trip; parse as IEEE double).
Viterbi notes for the port: initial V[0][y] = start(y) + emit(y, obs0) over
C[obs0] (or all states); step keeps prev states with non-empty T row, candidate
set = (C[obs] or all) & successors(prev), fallback to successors else all; ties
in max() break by comparing (prob, state-tuple) like Python tuple order.


NOTE: the posseg/finalseg HMM tables ship separately -- regenerate those with generate_zh_posseg.py
(the section this script once emitted used a different, incompatible layout).
"""
import gzip, io, os

OUT_DIR = r'C:\Users\lyrco\source\repos\MisakiSharp\data'
G2P_PATH = os.path.join(OUT_DIR, 'zh_g2p_v2.tsv.gz')
HMM_PATH = os.path.join(OUT_DIR, 'zh_posseg_hmm.tsv.gz')
MIN_FLOAT = -3.14e100

MISAKI_PHRASES = {
    '开户行': [['ka1i'], ['hu4'], ['hang2']],
    '发卡行': [['fa4'], ['ka3'], ['hang2']],
    '放款行': [['fa4ng'], ['kua3n'], ['hang2']],
    '茧行': [['jia3n'], ['hang2']],
    '行号': [['hang2'], ['ha4o']],
    '各地': [['ge4'], ['di4']],
    '借还款': [['jie4'], ['hua2n'], ['kua3n']],
    '时间为': [['shi2'], ['jia1n'], ['we2i']],
    '为准': [['we2i'], ['zhu3n']],
    '色差': [['se4'], ['cha1']],
    '嗲': [['dia3']],
    '呗': [['bei5']],
    '不': [['bu4']],
    '咗': [['zuo5']],
    '嘞': [['lei5']],
    '掺和': [['chan1'], ['huo5']],
}


def init_pypinyin():
    from pypinyin import load_phrases_dict, load_single_dict
    from pypinyin_dict.phrase_pinyin_data import large_pinyin
    large_pinyin.load()
    load_phrases_dict(MISAKI_PHRASES)
    load_single_dict({ord('地'): 'de,di4'})


def gz_writer(path):
    raw = open(path, 'wb')
    gz = gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=9)
    return io.TextIOWrapper(gz, encoding='utf-8', newline='\n')


def build_g2p():
    from pypinyin import lazy_pinyin, Style
    from pypinyin.constants import PHRASES_DICT, PINYIN_DICT
    in_range = lambda s: all('\u4e00' <= c <= '\u9fff' for c in s)
    syl_map, entries = {}, {}

    def add(key):
        kw = dict(neutral_tone_with_five=True)
        t3 = lazy_pinyin(key, style=Style.TONE3, **kw)
        ini = lazy_pinyin(key, style=Style.INITIALS, **kw)
        fin = lazy_pinyin(key, style=Style.FINALS_TONE3, **kw)
        assert len(t3) == len(ini) == len(fin) == len(key), (key, t3, ini, fin)
        for s, i, f in zip(t3, ini, fin):
            assert s and '\t' not in s and ' ' not in s, (key, s)
            prev = syl_map.setdefault(s, (i, f))
            assert prev == (i, f), ('syl conflict', key, s, prev, (i, f))
        entries[key] = ' '.join(t3)

    chars = [chr(cp) for cp in sorted(PINYIN_DICT) if 0x4E00 <= cp <= 0x9FFF]
    for c in chars: add(c)
    phrases = sorted(k for k in PHRASES_DICT if len(k) >= 2 and in_range(k))
    for p in phrases: add(p)
    dropped = sorted(k for k in PHRASES_DICT if len(k) >= 2 and not in_range(k))
    print(f'chars: {len(chars)}  phrases: {len(phrases)}  dropped (out of 一-鿿): {len(dropped)}')
    print('dropped sample:', dropped[:8])
    print(f'distinct TONE3 syllables: {len(syl_map)}')
    return syl_map, chars, phrases, entries


def load_jieba():
    import jieba
    jieba.initialize()
    words = {}
    with open(os.path.join(os.path.dirname(jieba.__file__), 'dict.txt'), encoding='utf-8') as f:
        for line in f:
            w, freq, pos = line.split()
            words[w] = (int(freq), pos)
    from_freq = {w: f for w, f in jieba.dt.FREQ.items() if f > 0}
    assert from_freq == {w: fp[0] for w, fp in words.items()}, 'dict.txt parse != jieba.dt.FREQ'
    print(f'jieba words: {len(words)}  total: {jieba.dt.total}')
    return words, jieba.dt.total


def write_g2p(syl_map, chars, phrases, entries, words, total):
    with gz_writer(G2P_PATH) as f:
        for s in sorted(syl_map):
            f.write(f'#{s}\t{syl_map[s][0]},{syl_map[s][1]}\n')
        for k in chars: f.write(f'{k}\t{entries[k]}\n')
        for k in phrases: f.write(f'{k}\t{entries[k]}\n')
        f.write(f'@@\t{total}\n')
        for w, (fr, pos) in words.items():
            assert '\t' not in w and ' ' not in w, w
            f.write(f'@{w}\t{fr} {pos}\n')


def write_hmm():
    from jieba.posseg import char_state_tab, prob_start, prob_trans, prob_emit
    import jieba.posseg.viterbi as vt
    assert getattr(vt, 'MIN_FLOAT', MIN_FLOAT) == MIN_FLOAT
    states = sorted(prob_trans.P.keys())
    idx = {s: i for i, s in enumerate(states)}
    assert set(prob_start.P) == set(states) and set(prob_emit.P) <= set(states)
    assert all(s in idx for tup in char_state_tab.P.values() for s in tup)
    with gz_writer(HMM_PATH) as f:
        for i, (bmes, pos) in enumerate(states):
            f.write(f'S\t{i}\t{bmes},{pos}\n')
        for s in states:
            if (lp := prob_start.P[s]) != MIN_FLOAT: f.write(f'P\t{idx[s]}\t{lp!r}\n')
        for s in states:
            row = prob_trans.P[s]
            if row: f.write(f'T\t{idx[s]}\t' + ' '.join(f'{idx[t]}:{lp!r}' for t, lp in sorted(row.items(), key=lambda kv: idx[kv[0]])) + '\n')
        for s in states:
            row = prob_emit.P.get(s)
            if not row: continue
            assert all(len(ch) == 1 and ch not in ': \t' for ch in row)
            f.write(f'E\t{idx[s]}\t' + ' '.join(f'{ch}:{lp!r}' for ch, lp in sorted(row.items())) + '\n')
        for ch in sorted(char_state_tab.P):
            assert len(ch) == 1 and ch not in ': \t'
            f.write(f'C\t{ch}\t' + ' '.join(str(idx[s]) for s in sorted(char_state_tab.P[ch], key=idx.get)) + '\n')
    print(f'HMM states: {len(states)}  emit entries: {sum(len(v) for v in prob_emit.P.values())}  chars: {len(char_state_tab.P)}')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    init_pypinyin()
    syl_map, chars, phrases, entries = build_g2p()
    words, total = load_jieba()
    write_g2p(syl_map, chars, phrases, entries, words, total)
    write_hmm()
    for p in (G2P_PATH, HMM_PATH):
        print(f'{p}: {os.path.getsize(p):,} bytes gz')


if __name__ == '__main__':
    main()
