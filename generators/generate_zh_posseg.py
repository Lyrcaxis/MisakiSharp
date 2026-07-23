# Exports jieba.posseg's dict + HMM tables into the coordinate TSV formats PossegTagger.cs consumes.
# Outputs (UTF-8, gzip):
#   zh_posseg_dict.tsv.gz : "@@\t<total>" line, then "@<word>\t<freq> <pos>" lines (last dict-file occurrence wins, like jieba).
#   zh_posseg_hmm.tsv.gz  : "s\t<BMES>\t<pos>\t<logp>"                       (prob_start, all 256 states)
#                           "t\t<BMES>\t<pos>\t<BMES2>\t<pos2>\t<logp>"      (prob_trans)
#                           "e\t<BMES>\t<pos>\t<char>\t<logp>"               (prob_emit)
#                           "c\t<char>\t<BMES>:<pos>,<BMES>:<pos>,..."       (char_state_tab, order preserved)
# Floats are repr() shortest round-trip strings; freqs stay integers (C# recomputes Math.Log).
# log_check.tsv: "<freq>\t<repr(math.log(freq))>" for every distinct freq + total, so the C# harness can prove Math.Log == math.log.
import gzip, math, os

import jieba
import jieba.posseg as psg
from jieba.posseg import char_state_tab_P, start_P, trans_P, emit_P

jieba.initialize()
out_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))

words, tags = {}, {}
with open(jieba.dt.get_dict_file().name, 'rb') as f:
    for line in f:
        w, fr, tag = line.strip().decode('utf-8').split(' ')
        words[w] = int(fr)
        tags[w] = tag
assert set(words) == set(psg.dt.word_tag_tab), 'dict words vs word_tag_tab mismatch'
assert all(tags[w] == psg.dt.word_tag_tab[w] for w in words), 'tag mismatch vs word_tag_tab'
assert all(fr > 0 for fr in words.values()), 'zero-freq dict word would break truthy checks'
assert not any('\t' in w for w in words), 'tab inside dict word'

with gzip.open(os.path.join(out_dir, 'zh_posseg_dict.tsv.gz'), 'wt', encoding='utf-8', newline='\n') as f:
    f.write('@@\t%d\n' % jieba.dt.total)
    for w, fr in words.items():
        f.write('@%s\t%d %s\n' % (w, fr, tags[w]))

assert not any(ord(c) < 0x21 for d in emit_P.values() for c in d), 'control char in emit table'
assert not any(ord(c) < 0x21 for c in char_state_tab_P), 'control char in char_state_tab'
with gzip.open(os.path.join(out_dir, 'zh_posseg_hmm.tsv.gz'), 'wt', encoding='utf-8', newline='\n') as f:
    for (b, pos), p in start_P.items():
        f.write('s\t%s\t%s\t%s\n' % (b, pos, repr(p)))
    for (b, pos), nexts in trans_P.items():
        for (b2, pos2), p in nexts.items():
            f.write('t\t%s\t%s\t%s\t%s\t%s\n' % (b, pos, b2, pos2, repr(p)))
    for (b, pos), emits in emit_P.items():
        for ch, p in emits.items():
            f.write('e\t%s\t%s\t%s\t%s\n' % (b, pos, ch, repr(p)))
    for ch, states in char_state_tab_P.items():
        f.write('c\t%s\t%s\n' % (ch, ','.join('%s:%s' % s for s in states)))

freqs = sorted(set(words.values()) | {jieba.dt.total})
with open(os.path.join(out_dir, 'log_check.tsv'), 'w', encoding='utf-8', newline='\n') as f:
    for fr in freqs:
        f.write('%d\t%s\n' % (fr, repr(math.log(fr))))

print('words:', len(words), 'total:', jieba.dt.total, 'distinct freqs:', len(freqs))
print('start:', len(start_P), 'trans pairs:', sum(len(v) for v in trans_P.values()),
      'emit pairs:', sum(len(v) for v in emit_P.values()), 'chars:', len(char_state_tab_P))


# jieba's plain-cut finalseg HMM (BMES only) -- MisakiSharp's cut_for_search needs it for out-of-dict runs
from jieba.finalseg.prob_start import P as fs_start
from jieba.finalseg.prob_trans import P as fs_trans
from jieba.finalseg.prob_emit import P as fs_emit
with gzip.open(os.path.join(out_dir, 'zh_finalseg_hmm.tsv.gz'), 'wt', encoding='utf-8', newline='
') as f:
    for st, p in fs_start.items(): f.write(f"s	{st}	{p!r}
")
    for st, d in fs_trans.items():
        for st2, p in d.items(): f.write(f"t	{st}	{st2}	{p!r}
")
    for st, d in fs_emit.items():
        for ch, p in d.items(): f.write(f"e	{st}	{ch}	{p!r}
")
print('wrote zh_finalseg_hmm.tsv.gz')
