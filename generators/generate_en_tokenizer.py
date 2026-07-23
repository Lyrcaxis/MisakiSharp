# Exports spacy's English tokenizer data (affix regexes translated to .NET syntax + special-case rules)
# into MisakiSharp/en_tokenizer.tsv.gz for the native C# EnglishTokenizer.
import gzip, io, os, re, sys
import spacy
from spacy.attrs import ORTH

OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "en_tokenizer.tsv.gz"))

HEX = "0123456789abcdefABCDEF"
CLASS_SHORTHANDS = set("wWdDsS")
SINGLE_ESCAPES = {"t": 0x09, "n": 0x0A, "r": 0x0D, "f": 0x0C, "v": 0x0B, "a": 0x07, "b": 0x08, "0": 0x00}


def parse_class_items(body):
    """Parses a python-re character-class body into items: ('cp', int), ('range', lo, hi), ('esc', text)."""
    items, i, n = [], 0, len(body)
    def read_one(j):
        if body[j] == "\\":
            e = body[j + 1]
            if e == "u": return int(body[j + 2:j + 6], 16), j + 6
            if e == "U": return int(body[j + 2:j + 10], 16), j + 10
            if e == "x": return int(body[j + 2:j + 4], 16), j + 4
            if e in CLASS_SHORTHANDS: return ("esc", "\\" + e), j + 2
            if e in SINGLE_ESCAPES: return SINGLE_ESCAPES[e], j + 2
            return ord(e), j + 2
        return ord(body[j]), j + 1
    while i < n:
        cp, i = read_one(i)
        if isinstance(cp, tuple): items.append(cp); continue
        if i < n and body[i] == "-" and i + 1 < n:
            hi, i2 = read_one(i + 1)
            if isinstance(hi, tuple): raise ValueError("range to shorthand")
            items.append(("range", cp, hi)); i = i2
        else:
            items.append(("cp", cp))
    return items


def esc_bmp(cp):
    if 0x30 <= cp <= 0x39 or 0x41 <= cp <= 0x5A or 0x61 <= cp <= 0x7A: return chr(cp)
    return "\\u%04X" % cp


def bmp_range(lo, hi):
    """Emits a BMP range, cutting out the surrogate block -- .NET matches per UTF-16 unit, so lone halves must never match."""
    if hi >= 0xD800 and lo <= 0xDFFF: return bmp_range(lo, 0xD7FF) if hi <= 0xDFFF else bmp_range(0xE000, hi) if lo >= 0xD800 else bmp_range(lo, 0xD7FF) + bmp_range(0xE000, hi)
    if lo > hi: return ""
    return esc_bmp(lo) if lo == hi else esc_bmp(lo) + "-" + esc_bmp(hi)


def astral_alts(lo, hi):
    """Expands an astral codepoint range into .NET surrogate-pair alternatives (head pair, full-low middle block, tail pair)."""
    pair = lambda cp: (0xD800 + ((cp - 0x10000) >> 10), 0xDC00 + ((cp - 0x10000) & 0x3FF))
    (h1, l1), (h2, l2) = pair(lo), pair(hi)
    def alt(h, la, lb):
        head = "\\u%04X" % h[0] if h[0] == h[1] else "[\\u%04X-\\u%04X]" % h
        return head + ("\\u%04X" % la if la == lb else "[\\u%04X-\\u%04X]" % (la, lb))
    if h1 == h2: return [alt((h1, h1), l1, l2)]
    alts = [] if l1 == 0xDC00 else [alt((h1, h1), l1, 0xDFFF)]
    mid_lo = h1 if l1 == 0xDC00 else h1 + 1
    mid_hi = h2 if l2 == 0xDFFF else h2 - 1
    if mid_lo <= mid_hi: alts.append(alt((mid_lo, mid_hi), 0xDC00, 0xDFFF))
    if l2 != 0xDFFF: alts.append(alt((h2, h2), 0xDC00, l2))
    return alts


def word_char_sets():
    """Scans python's \\w over all codepoints (its class semantics differ from .NET's), returning exact BMP class text + astral alternatives."""
    word = re.compile(r"\w", re.UNICODE)
    runs, run_start, prev = [], None, None
    for cp in range(0x110000):
        if word.match(chr(cp)):
            if run_start is None: run_start = cp
            prev = cp
        elif run_start is not None:
            runs.append((run_start, prev)); run_start = None
    if run_start is not None: runs.append((run_start, prev))
    bmp, alts = [], []
    for lo, hi in runs:
        if lo <= 0xFFFF: bmp.append(bmp_range(lo, min(hi, 0xFFFF)))
        if hi > 0xFFFF: alts += astral_alts(max(lo, 0x10000), hi)
    return "".join(bmp), alts


def convert_class(pattern, start):
    """Converts one [...] class at pattern[start] into .NET-safe form, returning (text, next_index)."""
    assert pattern[start] == "["
    i, n = start + 1, len(pattern)
    assert pattern[i] != "^", "negated classes unsupported"
    body = []
    while i < n:
        if pattern[i] == "\\": body.append(pattern[i:i + 2]); i += 2; continue
        if pattern[i] == "]": break
        body.append(pattern[i]); i += 1
    assert i < n, "unterminated class"
    items = parse_class_items("".join(body))
    bmp, alts = [], []
    for it in items:
        if it[0] == "esc":
            if it[1] == "\\w": bmp.append(WORD_BMP); alts += WORD_ALTS
            else: bmp.append(it[1])
        elif it[0] == "cp":
            if it[1] <= 0xFFFF: bmp.append(bmp_range(it[1], it[1]))
            else: alts += astral_alts(it[1], it[1])
        else:
            lo, hi = it[1], it[2]
            if lo <= 0xFFFF: bmp.append(bmp_range(lo, min(hi, 0xFFFF)))
            if hi > 0xFFFF: alts += astral_alts(max(lo, 0x10000), hi)
    cls = "[" + "".join(bmp) + "]" if bmp else ""
    if not alts: return cls, i + 1
    return "(?:" + "|".join(([cls] if cls else []) + alts) + ")", i + 1


def to_dotnet(pattern):
    out, i, n = [], 0, len(pattern)
    if pattern.startswith("(?u)"): i = 4
    while i < n:
        c = pattern[i]
        if c == "\\":
            e = pattern[i + 1]
            if e == "U":
                cp = int(pattern[i + 2:i + 10], 16)
                out.append("\\u%04X\\u%04X" % (0xD800 + ((cp - 0x10000) >> 10), 0xDC00 + ((cp - 0x10000) & 0x3FF)))
                i += 10
            else: out.append(pattern[i:i + 2]); i += 2
        elif c == "[":
            cls, i = convert_class(pattern, i)
            out.append(cls)
        else:
            assert ord(c) <= 0xFFFF, "literal astral outside class"
            out.append(c); i += 1
    return "".join(out)


def esc_field(s):
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")


WORD_BMP, WORD_ALTS = word_char_sets()


def main():
    nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "ner", "attribute_ruler", "lemmatizer", "tok2vec", "senter"])
    tok = nlp.tokenizer
    assert tok.token_match is None and tok.faster_heuristics
    lines = []
    for sigil, fn in [("P", tok.prefix_search), ("S", tok.suffix_search), ("I", tok.infix_finditer), ("U", tok.url_match)]:
        pat = fn.__self__.pattern
        assert fn.__self__.flags == re.UNICODE and "\t" not in pat and "\n" not in pat
        lines.append(sigil + "\t" + to_dotnet(pat))
    for chunk, substrings in sorted(tok.rules.items()):
        lines.append("@" + esc_field(chunk) + "\t" + "\t".join(esc_field(s[ORTH]) for s in substrings))
    with gzip.open(OUT, "wb") as f:
        f.write(("\n".join(lines) + "\n").encode("utf-8"))
    print(f"wrote {OUT}: 4 patterns + {len(tok.rules)} rules")


if __name__ == "__main__":
    main()
