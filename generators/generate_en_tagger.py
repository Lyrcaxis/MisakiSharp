# Distills spacy en_core_web_sm's Penn Treebank tagger into a portable averaged perceptron for the
# C# EnglishTagger port (Processing/EnglishTagger.cs). Run stages in order (or 'all'):
#
#   python generate_en_tagger.py corpus    -> parity/en_tagger_corpus.tsv.gz   (cat \t escaped sentence)
#   python generate_en_tagger.py tag       -> parity/en_tagger_tagged.tsv.gz   (cat \t json tokens \t json spacy tags)
#   python generate_en_tagger.py train     -> parity/en_tagger_weights.pkl.gz  (float averaged weights + tagdict)
#   python generate_en_tagger.py export    -> KokoroSharp/Processing/en_tagger.bin.gz (quantized, pruned)
#   python generate_en_tagger.py fixtures  -> parity/en_tagger_fixtures.tsv    (held-out: json tokens \t json OWN-model tags)
#   python generate_en_tagger.py parity    -> tag agreement + misaki phoneme-level parity report (stdout)
#
# Held-out protocol: eval = md5(sentence) % 20 == 0 (5%), never trained on. All reported numbers are
# on that eval split (or on the reserved en_fixtures.tsv sentences, which are excluded from training too).
#
# Bit-exactness with C#: every feature string is built from parity/en_csharp_chartable.tsv.gz (dumped
# by .NET itself: per BMP unit -> char.ToLowerInvariant, char.IsUpper, char.IsLower), astral chars fold
# to two U+FFFD (mirroring C#'s per-UTF-16-unit processing), digits '0'-'9' fold to '0', and argmax
# tie-breaks by ordinal tag comparison. Fixture tags come from the binary as read back from disk.
#
# en_tagger.bin.gz format (gzip; strings are BinaryWriter-style: 7-bit-encoded byte length + UTF-8):
#   string "KTAG1"
#   byte tagCount, tagCount x string tag                      (kept in written order = py sorted())
#   varint tagdictCount, x (string rawWord, byte tagIdx)      (frequent unambiguous words skip scoring)
#   varint featureCount, x (string key, byte pairCount, pairCount x (byte tagIdx, varint zigzag(weight)))
import gzip, hashlib, json, math, os, pickle, random, re, sys, time, urllib.request
from collections import Counter, defaultdict

BIN = os.path.dirname(os.path.abspath(__file__))
PARITY = os.path.join(BIN, 'models_notpushed', 'parity')
MISAKI = os.path.join(BIN, 'models_notpushed', 'misaki')
DEMO = os.path.join(BIN, 'models_notpushed', 'kokoro', 'demo')
OUT_BIN = r'C:\Users\lyrco\source\repos\MisakiSharp\data\en_tagger.bin.gz'
sys.stdout.reconfigure(encoding='utf-8')

# ---------------------------------------------------------------- C#-exact char ops
_rows = gzip.open(os.path.join(PARITY, 'en_csharp_chartable.tsv.gz'), 'rt', encoding='utf-8').read().splitlines()
CS_LOWER = [int(r.split('\t')[0]) for r in _rows]
CS_UPPER = [r.split('\t')[1][0] == '1' for r in _rows]
CS_ISLOW = [r.split('\t')[1][1] == '1' for r in _rows]

def norm(word):  # C#: per UTF-16 unit -> ToLowerInvariant, ascii digit -> '0', surrogate unit -> U+FFFD
    out = []
    for ch in word:
        cp = ord(ch)
        if cp > 0xFFFF: out.append('\ufffd\ufffd'); continue
        c = chr(CS_LOWER[cp])
        out.append('0' if '0' <= c <= '9' else c)
    return ''.join(out)

def shape(word):  # flags over raw units: U all-cased-upper, F first-upper, D has ascii digit, H has '-'
    units = []
    for ch in word:
        cp = ord(ch)
        units.extend((0xFFFD, 0xFFFD) if cp > 0xFFFF else (cp,))
    cased = [u for u in units if u <= 0xFFFF and (CS_UPPER[u] or CS_ISLOW[u])]
    flags = ('U' if cased and all(CS_UPPER[u] for u in cased) else '') \
          + ('F' if units and units[0] <= 0xFFFF and CS_UPPER[units[0]] else '') \
          + ('D' if any(0x30 <= u <= 0x39 for u in units) else '') + ('H' if 0x2D in units else '')
    return flags or '-'

def suf(s, k): return s if len(s) <= k else s[-k:]

def features(i, norms, shapes, prev, prev2):
    w, pw, qw, nw, ow = norms[i + 2], norms[i + 1], norms[i], norms[i + 3], norms[i + 4]
    return ('b', 'w=' + w, 's1=' + suf(w, 1), 's2=' + suf(w, 2), 's3=' + suf(w, 3), 'p1=' + w[:1],
            'sh=' + shapes[i], 't1=' + prev, 't2=' + prev2, 'tt=' + prev2 + '|' + prev, 'tw=' + prev + '|' + w,
            'pw=' + pw, 'ps=' + suf(pw, 3), 'qw=' + qw, 'nw=' + nw, 'ns=' + suf(nw, 3), 'ow=' + ow)

def pad_norms(tokens): return ['<S2>', '<S>'] + [norm(t) for t in tokens] + ['<E>', '<E2>']

def is_eval(sentence): return int(hashlib.md5(sentence.encode('utf-8')).hexdigest(), 16) % 20 == 0

def esc(s): return s.replace('\\', '\\\\').replace('\t', '\\t').replace('\n', '\\n')
def unesc(s): return s.replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')

# ---------------------------------------------------------------- stage: corpus
BOOKS = [1342, 2701, 1661, 11, 84, 345, 98, 74, 76, 36, 174, 158, 1400, 219, 43, 35, 5200, 205,
         2554, 844, 1232, 120, 55, 46, 768, 1260, 829, 2591, 1080, 16]
ABBREV = {'mr', 'mrs', 'ms', 'dr', 'st', 'vs', 'etc', 'no', 'prof', 'rev', 'col', 'gen', 'capt', 'lt',
          'hon', 'jr', 'sr', 'vol', 'ch', 'fig', 'op', 'ca', 'esq', 'messrs', 'mme', 'mlle'}

def fetch_book(book_id):
    cache = os.path.join(PARITY, 'gutenberg')
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f'{book_id}.txt')
    if not os.path.exists(path):
        for url in (f'https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt',
                    f'https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt'):
            try:
                data = urllib.request.urlopen(url, timeout=60).read().decode('utf-8-sig', errors='replace')
                open(path, 'w', encoding='utf-8', newline='\n').write(data)
                time.sleep(0.5)
                break
            except Exception as e:
                print(f'  {book_id}: {url} failed ({e})')
        else:
            return ''
    return open(path, encoding='utf-8').read()

def split_sentences(par):
    out, start = [], 0
    for m in re.finditer(r'[.!?…]+[\'"”’)\]]*\s+', par):
        nxt = par[m.end():m.end() + 1]
        if not nxt or not (nxt.isupper() or nxt.isdigit() or nxt in '“"\'‘(['): continue
        lastword = par[start:m.start()].rsplit(None, 1)[-1].lstrip('“"\'‘([') if par[start:m.start()].split() else ''
        if lastword.rstrip('.').lower() in ABBREV or (len(lastword) == 1 and lastword.isupper()): continue
        out.append(par[start:m.end()].strip())
        start = m.end()
    out.append(par[start:].strip())
    return [s for s in out if s]

def book_sentences(text):
    m = re.search(r'\*\*\* ?START OF.*?\*\*\*', text)
    if m: text = text[m.end():]
    m = re.search(r'\*\*\* ?END OF', text)
    if m: text = text[:m.start()]
    sents = []
    for par in re.split(r'\n\s*\n', text.replace('_', '')):
        par = re.sub(r'\s+', ' ', par).strip()
        if not par or not any(c.isalpha() for c in par): continue
        sents.extend(s for s in split_sentences(par) if 2 <= len(s.split()) <= 60)
    return sents

def synthetic_corpus(rng):
    corpus = []
    def add(cat, sents): corpus.extend((cat, s) for s in sents)

    homographs = ['read', 'lead', 'live', 'wind', 'bass', 'bow', 'tear', 'close', 'record', 'present',
                  'object', 'minute', 'contract', 'produce', 'conduct', 'content', 'desert', 'subject',
                  'permit', 'project', 'rebel', 'suspect', 'convert', 'increase', 'insult', 'protest',
                  'console', 'invalid', 'moderate', 'separate', 'graduate', 'estimate', 'associate',
                  'deliberate', 'alternate', 'duplicate', 'advocate', 'excuse', 'abuse', 'house', 'use',
                  'refuse', 'dove', 'wound', 'sow', 'row', 'resume', 'perfect', 'frequent', 'attribute',
                  'compound', 'conflict', 'contest', 'decrease', 'defect', 'discount', 'escort', 'exploit',
                  'export', 'extract', 'impact', 'import', 'incline', 'progress', 'transfer', 'transplant',
                  'upset', 'addict', 'combine', 'compress', 'conscript', 'convict', 'digest', 'entrance',
                  'intern', 'misuse', 'perfume', 'proceeds', 'rerun', 'segment', 'survey', 'torment']
    verb_ctx = ["They will {w} it tomorrow.", "Please {w} the papers now.", "He wants to {w} again soon.",
                "She decided to {w} the offer.", "Did you {w} it last night?", "We {w} every single day.",
                "I {w} it yesterday morning.", "Don't {w} anything yet.", "You should {w} before noon.",
                "Who would {w} such a thing?", "Let me {w} this first.", "They might {w} the whole batch.",
                "He refused to {w} the shipment.", "Can we {w} it together?", "She will not {w} the plan.",
                "To {w} or not, that is the question.", "We must {w} carefully.", "I have to {w} the results.",
                "Watch him {w} the crowd.", "Nobody dared to {w} it."]
    noun_ctx = ["The {w} was impressive to see.", "A {w} appeared near the door.", "Its {w} surprised everyone.",
                "That {w} belongs to me.", "Every {w} counts in the end.", "His {w} broke last week.",
                "What a strange {w} that was.", "The old {w} still works fine.", "Another {w} arrived today.",
                "This {w} costs too much.", "Her {w} won the prize.", "Each {w} was carefully labeled.",
                "The {w} on the table is mine.", "No {w} lasts forever.", "Their {w} caused the delay.",
                "One {w} is never enough.", "Some {w} or other blocked the road.", "The {w} itself seemed fine.",
                "Behind the {w} stood a clerk.", "My {w} needs repairs."]
    for w in homographs:
        for t in rng.sample(verb_ctx, 9) + rng.sample(noun_ctx, 9):
            add('homograph', [t.format(w=w)])

    add('homograph_pairs', [
        "I read the book yesterday, but she will read it tomorrow.",
        "He has read every novel that I read aloud.", "The lead singer wore a lead apron.",
        "Follow my lead and lead the way.", "I live near a live wire, and we watched it live.",
        "The wind will wind around the tower.", "Please wind the clock before the wind picks up.",
        "He caught a bass while playing bass.", "She took a bow after tying the bow on the bow of the ship.",
        "The dove dove down from the roof.", "A single tear rolled down as he began to tear the letter.",
        "The nurse wound gauze around the wound.", "They refuse to collect the refuse on Sundays.",
        "I present you this present, and he will present it later.",
        "For the record, they record every record attempt.", "The farm can produce fresh produce all year.",
        "The conductor will conduct with impeccable conduct.", "I am content with the content of this book.",
        "Don't desert me in the desert.", "The object of the game is to object loudly.",
        "The subject refused to subject himself to more tests.", "You need a permit to permit such behavior.",
        "The project team will project the numbers on screen.", "The rebel decided to rebel again.",
        "I suspect the suspect is lying.", "The convert tried to convert his friends.",
        "Please contract the lawyer before you sign the contract.", "Prices increase when demand shows an increase.",
        "Wait a minute, this detail is quite minute.", "Close the door, it's too close to the street.",
        "The dog was used to being used to hunt.", "She used to sing, and he used the microphone.",
    ])

    companies = ['Acme Corp', 'Google', 'Microsoft', 'Boeing', 'Tesla', 'Walmart', 'Amazon', 'Apple',
                 'Netflix', 'Intel', 'IBM', 'Ford', 'Toyota', 'Samsung', 'Siemens', 'Shell', 'Pfizer',
                 'Airbus', 'Sony', 'Nvidia', 'Oracle', 'Chevron', 'Nike', 'Visa', 'Disney']
    orgs = ['the Senate', 'Congress', 'the Fed', 'the EU', 'the UN', 'the White House', 'parliament',
            'the ministry', 'the city council', 'the board', 'regulators', 'the court', 'the union']
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
              'October', 'November', 'December']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    news_t = ["{c} announced on {day} that revenue rose {n}% to ${m} billion in Q{q} {y}.",
              "{c} shares fell {n}% after the company reported weaker guidance.",
              "{o} approved the merger between {c} and {c2} on {mo} {d}, {y}.",
              "Analysts expect {c} to cut about {m},000 jobs by {mo} {y}.",
              "{c} said it will invest ${m} million in a new plant near {city}.",
              "{o} warned that inflation could reach {n}.{n2}% by late {y}.",
              "The storm hit {city} on {day} night, leaving {m},000 homes without power.",
              "{city} beat {city2} {sc1}-{sc2} in overtime on {day}.",
              "A jury found the defendant guilty on {d} of {d2} counts.",
              "{c}'s CEO told {o} that the recall affects {m},000 vehicles.",
              "Unemployment dropped to {n}.{n2}% in {mo}, the lowest since {y}.",
              "{o} voted {sc1}-{sc2} to raise the minimum wage to ${n}.{n2} an hour.",
              "Oil prices climbed {n}% to ${m} a barrel on {day}.",
              "{c} unveiled its new model at a {city} event, priced from ${m},999."]
    cities = ['Chicago', 'Boston', 'Denver', 'Seattle', 'Atlanta', 'Dallas', 'Miami', 'Portland',
              'London', 'Paris', 'Berlin', 'Tokyo', 'Madrid', 'Toronto', 'Austin', 'Phoenix']
    for _ in range(3200):
        c, c2 = rng.sample(companies, 2)
        city, city2 = rng.sample(cities, 2)
        add('news', [rng.choice(news_t).format(
            c=c, c2=c2, o=rng.choice(orgs), day=rng.choice(days), mo=rng.choice(months), city=city, city2=city2,
            n=rng.randrange(1, 30), n2=rng.randrange(0, 10), m=rng.randrange(1, 900), q=rng.randrange(1, 5),
            y=rng.randrange(1998, 2027), d=rng.randrange(1, 29), d2=rng.randrange(3, 30),
            sc1=rng.randrange(0, 120), sc2=rng.randrange(0, 120))])

    names = ['Tom', 'Anna', 'Mrs. Hale', 'Dr. Reed', 'Sam', 'Ellie', 'the captain', 'her mother',
             'Mr. Brooks', 'Grandpa', 'the stranger', 'Nora', 'Ben', 'Professor Lang', 'Maya', 'Jack']
    quips = ["I never asked for this", "we should leave before dark", "the bridge is out",
             "you can't be serious", "dinner is almost ready", "nothing ever changes here",
             "that's the third time this week", "someone has to tell him", "it wasn't me",
             "the letter arrived too late", "hold the door", "they'll never believe us"]
    dialog_t = ['"{q}," said {n}.', '"{q}!" shouted {n}.', '{n} whispered, "{q}."',
                '"{q}?" asked {n}, frowning.', '"{q}," {n} replied, "and {q2}."',
                "{n} asked whether {q3}.", "Why would {n} ever say that?", "\"Well,\" said {n}, \"{q}.\""]
    for _ in range(2400):
        q, q2 = rng.sample(quips, 2)
        add('dialog', [rng.choice(dialog_t).format(q=q, q2=q2, q3=q.replace('we ', 'they '), n=rng.choice(names))])

    add('questions', [
        "Are you coming to the party, or not?", "You're staying, aren't you?", "Didn't she warn you twice?",
        "What time does the last train leave?", "How much would it cost to repaint the fence?",
        "Whose coat is hanging by the door?", "Which of these routes is faster?", "Won't they mind the noise?",
        "Could you have known about the delay?", "Where were you on the night of the 12th?",
        "Isn't it odd that nobody called?", "Do you ever wonder why we bother?", "May I borrow your pen?",
        "Shall we begin, ladies and gentlemen?", "Have the results been posted yet?",
        "Who let the dogs out of the yard?", "Why not try the smaller hammer first?",
    ] * 8)

    gold = json.load(open(os.path.join(MISAKI, 'misaki', 'data', 'us_gold.json'), encoding='utf-8'))
    silver = json.load(open(os.path.join(MISAKI, 'misaki', 'data', 'us_silver.json'), encoding='utf-8'))
    gold_words = [k for k in gold if k.isalpha() and k == k.lower() and len(k) > 2]
    silver_words = [k for k in silver if k.isalpha() and k == k.lower() and len(k) > 2]
    salad_t = ["Words like {a}, {b}, and {c} appear often.", "The {a} was more {b} than the {c}.",
               "Never {a} a {b} without some {c}.", "His {a} seemed {b} to the {c}.",
               "A {a} of {b} lay beside the {c}.", "They tried to {a} the {b} near the {c}.",
               "Something {a} about the {b} made the {c} tremble.", "It was {a}, {b}, and utterly {c}."]
    for words, cat, count in ((gold_words, 'gold_salad', 5200), (silver_words, 'silver_salad', 3200)):
        for _ in range(count):
            a, b, c = rng.sample(words, 3)
            add(cat, [rng.choice(salad_t).format(a=a, b=b, c=c)])

    num_t = ["There are {n} items left.", "He counted {n} of them.", "Chapter {n} begins here.",
             "It measures {n} meters across.", "{n} people attended.", "Add {n} to the total.",
             "Row {n} is empty.", "The odometer read {n}."]
    nums = [str(rng.randrange(10 ** (mag - 1), 10 ** mag)) for mag in (1, 2, 3, 4, 5, 6, 7, 9, 12) for _ in range(40)]
    nums += [f'{rng.randrange(1, 10 ** 6):,}' for _ in range(120)]
    nums += [f'{rng.randrange(0, 1000)}.{rng.randrange(0, 99):02d}' for _ in range(90)]
    nums += [f'-{rng.randrange(1, 500)}' for _ in range(40)] + ['0', '00', '007', '0.001', '10.10.10', '1.2.3.4']
    for n in nums: add('numbers', [rng.choice(num_t).format(n=n)])
    for _ in range(220):
        y, y2 = rng.randrange(1066, 2100), rng.randrange(1500, 2050)
        add('years', [rng.choice(["In {y} everything changed.", "The {y} harvest failed.",
            "By {y}, nobody remembered {y2}.", "From {y2} to {y}, little happened.", "The {y}s were louder."]).format(y=y, y2=y2)])
    for _ in range(200):
        n = rng.randrange(1, 400)
        s = 'th' if n % 100 in (11, 12, 13) else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        add('ordinals', [rng.choice(["The {o} time was the charm.", "She finished {o} overall.",
            "On the {o} day, it rained.", "His {o} attempt succeeded.", "Turn left at the {o} light."]).format(o=f'{n}{s}')])
    for _ in range(260):
        amt = rng.choice([f'{rng.randrange(1, 100)}', f'{rng.randrange(1, 1000)}.{rng.randrange(0, 99):02d}',
                          f'{rng.randrange(1, 10 ** 6):,}', f'0.{rng.randrange(1, 99):02d}'])
        add('currency', [rng.choice(["It costs {c}{a} today.", "He paid {c}{a} for it.", "The fee rose to {c}{a}.",
            "{c}{a} seemed fair.", "They quoted {c}{a} per unit."]).format(c=rng.choice('$£€'), a=amt)])
    for _ in range(200):
        h, mi = rng.randrange(0, 24), rng.randrange(0, 60)
        add('times', [rng.choice(["The train leaves at {h}:{m:02d}.", "It's {h}:{m:02d} somewhere.",
            "Wake me at {h}:{m:02d} sharp.", "The shop closes at {h}:{m:02d} tonight."]).format(h=h, m=mi)])
    for _ in range(240):
        add('dates', [rng.choice(["The meeting is on {mo} {d}, {y}.", "She arrived on {d} {mo} {y}.",
            "Deadline: {mo}. {d}.", "Born {mo} {d}, died {mo2} {d2}.", "See you {day}, {mo} {d}th."]).format(
            mo=rng.choice(months), mo2=rng.choice(months), d=rng.randrange(1, 29), d2=rng.randrange(1, 29),
            y=rng.randrange(1800, 2027), day=rng.choice(days))])
    add('phones_versions', [f"Call {rng.randrange(200, 999)}-{rng.randrange(1000, 9999)} after five." for _ in range(60)]
        + [f"Version {rng.randrange(0, 12)}.{rng.randrange(0, 20)}.{rng.randrange(0, 20)} fixed the bug." for _ in range(60)]
        + [f"He scored {rng.randrange(1, 100)}% on the {rng.choice(['quiz', 'final', 'retest'])}." for _ in range(60)]
        + [f"Pages {a}-{a + rng.randrange(2, 60)} cover the war." for a in rng.sample(range(1, 400), 60)])

    subj = ["I", "You", "He", "She", "We", "They", "It"]
    add('contractions', [f"{s}'ll finish it by noon." for s in subj] + [f"{s}'d rather not say why." for s in subj]
        + [f"{s}'ve seen worse storms than this." for s in subj if s not in ('He', 'She', 'It')] + [
        "I'd've thought you'd know better.", "She's been where he's going.", "It'll rain, won't it?",
        "We're sure they're not on their way.", "You've got what I've lost.", "He'd rather she'd stay.",
        "That's what's wrong, ain't it?", "Who's to say whose fault it is?", "Let's see what's left.",
        "Couldn't you have? Shouldn't they? Wouldn't we?", "There's a there there, isn't there?",
        "What'll happen when it's done?", "Don't say I can't, because I won't.",
        "It's a shame they're not here, isn't it?", "Y'all shouldn't've come at eight o'clock.",
        "'Tis the season, ain't it?", "We've seen what they'll do when she'd rather leave.",
        "I'm not sure you're ready, but they've promised we'll cope.", "Mustn't grumble, needn't worry.",
        "The dog's leash and the dogs' bowls sat by James' car near John's house.",
    ] * 24)

    acr = ['NASA', 'FBI', 'IBM', 'USA', 'UK', 'EU', 'UN', 'CIA', 'DNA', 'URL', 'API', 'RAM', 'DVD',
           'GPS', 'ATM', 'PDF', 'SQL', 'XML', 'JSON', 'HTTP', 'FAQ', 'CEO', 'CFO', 'PhD', 'IQ', 'TV',
           'PC', 'NATO', 'WHO', 'GDP', 'MIT', 'BBC', 'NYSE', 'COVID-19', 'AT&T', 'U.S.', 'A.I.']
    for _ in range(560):
        a, b, c = rng.sample(acr, 3)
        add('acronyms', [rng.choice(["The {a} report cited the {b} and the {c}.", "{a} and {b} signed a deal with {c}.",
            "{a}'s budget dwarfs {b}'s.", "According to the {a}, {b} levels doubled.",
            "He left {a} for {b} after his {c} stint."]).format(a=a, b=b, c=c)])

    add('misc', [
        "THIS ENTIRE SENTENCE IS SHOUTED LOUDLY.", "BREAKING: MARKETS CLOSE EARLY TODAY.",
        "Local Hero Saves Cat From Tree", "Ten Ways To Boil An Egg Properly",
        "- first item\n- second item\n- third item", "# Heading\nSome text follows here.",
        "*emphasis* and **strong** text mix.", "`code_snippet()` returned null.",
        "Line one.\nLine two.\tTabbed three.", "Too  many   spaces    here.",
        "Visit example.com or www.google.com for details.", "Email me at user@example.com about the discount.",
        "The file is at C:/temp/file.txt on my PC.", "Oh! Well, um, yes, perhaps so.",
        "Hmm, no. Wait — yes! Absolutely not.", "Stop right there and put it down.",
        "Kindly forward the invoice at your earliest convenience.", "state-of-the-art, well-known, and half-baked ideas",
        "My mother-in-law uses snake_case and camelCase daily.", "The b2b startup pivoted to peer2peer sales.",
        "Rock 'n' roll never dies.", "O'Brien and O'Neill met D'Angelo.", "'Twas the night before Christmas.",
        "Mr. Smith met Dr. Jones on St. Patrick's Day.", "J.R.R. Tolkien wrote it, e.g. in 1954, i.e. long ago.",
        "He got his Ph.D. from MIT after his B.A.", "X-rays and MRI scans cost $450 each.",
        "The A-team drove a T-34 tank.", "Boeing 747s and A380s dwarf the 737.",
        "It's 5 o'clock somewhere, maybe 5:00 PM.", "AM radio at 6 AM plays jazz.",
        "One plus two equals three, minus four equals negative one.",
        "First, second, third, fourth, and finally fifth.", "A dozen eggs, a hundred cows, a thousand sheep.",
        "The naïve café charged Zoë five euros.", "He whispered zzyxwv and florptastic quietly.",
        "The glorbnak jumped over the frimble.", "She quixotted the brangleworth entirely.",
        "Judy read Tolstoy; Bob, Hemingway.", "The recipe (see page 12) needs salt; pepper, too!",
        "Wait — what? No… really?!", "She said “Hello there!” and left.",
        'He shouted "stop" — nobody listened…',
    ])
    wsmd_t = ["- {a} item\n- {b} item\n- {c} item", "# {A} report\nThe {b} section covers the {c}.",
              "{A} first.\n{B} second.\tThen the {c}.", "The {a}  had   two {b}s.\n\nA new {c} began.",
              "*{a}* and **{b}** met `{c}`.", "> {A} quotes the {b}.\n\nEnd of {c}.",
              "{A}, {b}, {c}:\n1. {a}\n2. {b}\n3. {c}"]
    gw = [w for w in gold_words if len(w) < 10]
    for _ in range(700):
        a, b, c = rng.sample(gw, 3)
        add('wsmd', [rng.choice(wsmd_t).format(a=a, b=b, c=c, A=a.capitalize(), B=b.capitalize())])
    for _ in range(500):
        a, b = rng.sample(gw, 2)
        add('questions2', [rng.choice(["Did the {a} ever {b}?", "Why is the {a} so {b}?", "Wasn't her {a} rather {b}?",
            "Can a {a} really {b}?", "Which {a} looked more {b}?", "How {b} was the {a}, honestly?"]).format(a=a, b=b)])
    for _ in range(500):
        a, b = rng.sample(gw, 2)
        add('contractions2', [rng.choice(["It's the {a} that won't {b}.", "They're {a}, but she's {b}.",
            "We'd have {b} if the {a} hadn't broken.", "You'll find the {a} isn't {b}.",
            "The {a}'s {b} wasn't anyone's fault.", "I've never seen a {a} {b} like that."]).format(a=a, b=b)])
    return corpus

def stage_corpus():
    rng = random.Random(20260724)
    reserved = set()
    with open(os.path.join(PARITY, 'en_fixtures.tsv'), encoding='utf-8') as f:
        for line in f:
            if line.startswith('S\t'): reserved.add(unesc(line.rstrip('\n').split('\t')[3]))

    corpus = []
    for book_id in BOOKS:
        sents = book_sentences(fetch_book(book_id))
        rng.shuffle(sents)
        corpus.extend((f'book{book_id}', s) for s in sents[:4200])
        print(f'  book {book_id}: {min(len(sents), 4200)} sentences')
    for line in open(os.path.join(DEMO, 'en.txt'), encoding='utf-8'):
        line = line.strip()
        if line: corpus.extend(('demo_en', s) for s in split_sentences(line))
    for name in ('frankenstein5k.md', 'gatsby5k.md'):
        corpus.extend((f'demo_{name[:5]}', s) for s in book_sentences(open(os.path.join(DEMO, name), encoding='utf-8').read()))
    corpus.extend(synthetic_corpus(rng))

    seen, out = set(reserved), []
    for cat, s in corpus:
        if s not in seen: seen.add(s); out.append((cat, s))
    rng.shuffle(out)
    with gzip.open(os.path.join(PARITY, 'en_tagger_corpus.tsv.gz'), 'wt', encoding='utf-8', newline='\n') as f:
        for cat, s in out: f.write(f'{cat}\t{esc(s)}\n')
    n_eval = sum(is_eval(s) for _, s in out)
    print(f'corpus: {len(out)} sentences ({n_eval} eval), {len(reserved)} reserved excluded')
    print('categories:', dict(Counter(c for c, _ in out).most_common()))

# ---------------------------------------------------------------- stage: tag
def stage_tag():
    import spacy
    nlp = spacy.load('en_core_web_sm', enable=['tok2vec', 'tagger'])
    rows = [line.rstrip('\n').split('\t') for line in gzip.open(os.path.join(PARITY, 'en_tagger_corpus.tsv.gz'), 'rt', encoding='utf-8')]
    texts = [unesc(s) for _, s in rows]
    t0, done = time.time(), 0
    with gzip.open(os.path.join(PARITY, 'en_tagger_tagged.tsv.gz'), 'wt', encoding='utf-8', newline='\n') as f:
        for (cat, _), doc in zip(rows, nlp.pipe(texts, batch_size=512)):
            toks, tags = [t.text for t in doc], [t.tag_ for t in doc]
            f.write(f'{cat}\t{json.dumps(toks, ensure_ascii=False)}\t{json.dumps(tags, ensure_ascii=False)}\n')
            done += 1
            if done % 20000 == 0: print(f'  tagged {done}/{len(texts)} ({time.time() - t0:.0f}s)')
    print(f'tagged {done} sentences in {time.time() - t0:.0f}s')

def load_tagged():
    train, eval_ = [], []
    for line in gzip.open(os.path.join(PARITY, 'en_tagger_tagged.tsv.gz'), 'rt', encoding='utf-8'):
        cat, toks, tags = line.rstrip('\n').split('\t')
        toks, tags = json.loads(toks), json.loads(tags)
        if not toks: continue
        (eval_ if is_eval(''.join(toks)) else train).append((cat, toks, tags))
    return train, eval_

# ---------------------------------------------------------------- stage: train
def build_tagdict(train):
    counts = defaultdict(Counter)
    for _, toks, tags in train:
        for w, t in zip(toks, tags): counts[w][t] += 1
    tagdict = {}
    for w, ctr in counts.items():
        tag, n = ctr.most_common(1)[0]
        if sum(ctr.values()) >= 20 and n / sum(ctr.values()) >= 0.99: tagdict[w] = tag
    return tagdict

def run_eval(eval_, tagdict, classes, score_fn):
    good = total = 0
    confusion = Counter()
    for _, toks, tags in eval_:
        guesses = predict_sentence(toks, tagdict, classes, score_fn)
        for g, t in zip(guesses, tags):
            good += g == t; total += 1
            if g != t: confusion[(t, g)] += 1
    return good / max(1, total), confusion

def predict_sentence(toks, tagdict, classes, score_fn):
    norms, shapes = pad_norms(toks), [shape(t) for t in toks]
    prev, prev2, out = '<S>', '<S2>', []
    for i, raw in enumerate(toks):
        guess = tagdict.get(raw)
        if guess is None:
            scores = defaultdict(float)
            for f in features(i, norms, shapes, prev, prev2): score_fn(f, scores)
            guess = classes[0]
            best = scores.get(guess, 0)
            for c in classes[1:]:
                s = scores.get(c, 0)
                if s > best or (s == best and c > guess): guess, best = c, s
        out.append(guess)
        prev2, prev = prev, guess
    return out

def stage_train():
    train, eval_ = load_tagged()
    print(f'train {len(train)} sentences, eval {len(eval_)} sentences')
    tagdict = build_tagdict(train)
    classes = sorted({t for _, _, tags in train for t in tags})
    print(f'tagdict {len(tagdict)} words, {len(classes)} classes: {classes}')

    sents = []
    for _, toks, tags in train:
        norms, shapes = pad_norms(toks), [shape(t) for t in toks]
        sents.append((toks, tags, norms, shapes))

    weights = {}  # feat -> {tag: [w, total, tstamp]}
    instances = 0
    rng = random.Random(31337)
    best_acc, drops = 0.0, 0
    for epoch in range(8):
        rng.shuffle(sents)
        correct = total = 0
        t0 = time.time()
        for toks, tags, norms, shapes in sents:
            prev, prev2 = '<S>', '<S2>'
            for i, raw in enumerate(toks):
                guess = tagdict.get(raw)
                if guess is None:
                    instances += 1
                    feats = features(i, norms, shapes, prev, prev2)
                    scores = defaultdict(float)
                    for f in feats:
                        fw = weights.get(f)
                        if fw:
                            for tag, entry in fw.items(): scores[tag] += entry[0]
                    guess, best = classes[0], scores.get(classes[0], 0)
                    for c in classes[1:]:
                        s = scores.get(c, 0)
                        if s > best or (s == best and c > guess): guess, best = c, s
                    truth = tags[i]
                    if guess != truth:
                        for f in feats:
                            fw = weights.setdefault(f, {})
                            for tag, delta in ((truth, 1.0), (guess, -1.0)):
                                entry = fw.setdefault(tag, [0.0, 0.0, 0])
                                entry[1] += (instances - entry[2]) * entry[0]
                                entry[0] += delta
                                entry[2] = instances
                    correct += guess == truth
                    total += 1
                prev2, prev = prev, guess
        averaged = {f: {t: (e[1] + (instances - e[2]) * e[0]) / instances for t, e in fw.items()}
                    for f, fw in weights.items()}
        def score_avg(f, scores, _a=averaged):
            fw = _a.get(f)
            if fw:
                for tag, w in fw.items(): scores[tag] += w
        acc, _ = run_eval(eval_, tagdict, classes, score_avg)
        print(f'epoch {epoch}: train-acc(nondict) {correct / max(1, total):.4f}, eval-acc(avg) {acc:.5f}, '
              f'{len(weights)} feats, {time.time() - t0:.0f}s')
        if acc <= best_acc + 0.00005: drops += 1
        else: best_acc, drops = acc, 0
        if drops >= 2: break

    with gzip.open(os.path.join(PARITY, 'en_tagger_weights.pkl.gz'), 'wb') as f:
        pickle.dump({'averaged': averaged, 'tagdict': tagdict, 'classes': classes}, f, protocol=4)
    print(f'saved float model: {len(averaged)} features, eval acc {best_acc:.5f}')

# ---------------------------------------------------------------- stage: export
def write_varint(out, n):
    while True:
        b = n & 0x7F; n >>= 7
        if n: out.append(b | 0x80)
        else: out.append(b); break

def write_str(out, s):
    data = s.encode('utf-8')
    write_varint(out, len(data)); out.extend(data)

def stage_export():
    with gzip.open(os.path.join(PARITY, 'en_tagger_weights.pkl.gz'), 'rb') as f:
        model = pickle.load(f)
    averaged, tagdict, classes = model['averaged'], model['tagdict'], model['classes']
    _, eval_ = load_tagged()

    max_w = max(abs(w) for fw in averaged.values() for w in fw.values())
    scale = next(s for s in (1000, 500, 200, 100, 50, 20) if max_w * s < 32000)
    print(f'max |w| {max_w:.2f} -> scale {scale}')

    def quantize(min_q):
        q = {}
        for f, fw in averaged.items():
            pairs = [(t, round(w * scale)) for t, w in fw.items()]
            pairs = [(t, w) for t, w in pairs if w != 0]
            if pairs and max(abs(w) for _, w in pairs) >= min_q: q[f] = dict(pairs)
        return q

    tag_idx = {t: i for i, t in enumerate(classes)}
    chosen = None
    for min_q in (1, 2, 3, 5, 8, 12):
        q = quantize(min_q)
        def score_int(f, scores, _q=q):
            fw = _q.get(f)
            if fw:
                for tag, w in fw.items(): scores[tag] += w
        acc, _ = run_eval(eval_, tagdict, classes, score_int)
        out = bytearray()
        write_str(out, 'KTAG1')
        out.append(len(classes))
        for t in classes: write_str(out, t)
        write_varint(out, len(tagdict))
        for w in sorted(tagdict): write_str(out, w); out.append(tag_idx[tagdict[w]])
        write_varint(out, len(q))
        for f in sorted(q):
            write_str(out, f)
            out.append(len(q[f]))
            for t in sorted(q[f]):
                out.append(tag_idx[t])
                v = q[f][t]
                write_varint(out, (v << 1) if v >= 0 else ((-v << 1) - 1))
        data = gzip.compress(bytes(out), 9)
        print(f'  min_q {min_q}: {len(q)} feats, eval acc {acc:.5f}, {len(data) / 1e6:.2f} MB gz')
        if chosen is None and len(data) < 14e6: chosen = (min_q, acc, data)
    min_q, acc, data = chosen
    open(OUT_BIN, 'wb').write(data)
    print(f'wrote {OUT_BIN}: min_q {min_q}, eval acc {acc:.5f}, {len(data) / 1e6:.2f} MB')

# ---------------------------------------------------------------- binary reader (the C# reference)
class BinModel:
    def __init__(self, path=OUT_BIN):
        data = gzip.open(path, 'rb').read()
        self.pos = 0
        def rv():
            n = shift = 0
            while True:
                b = data[self.pos]; self.pos += 1
                n |= (b & 0x7F) << shift; shift += 7
                if not b & 0x80: return n
        def rs():
            ln = rv(); s = data[self.pos:self.pos + ln].decode('utf-8'); self.pos += ln; return s
        assert rs() == 'KTAG1'
        n_tags = data[self.pos]; self.pos += 1
        self.classes = [rs() for _ in range(n_tags)]
        self.tagdict = {}
        for _ in range(rv()):
            w = rs(); self.tagdict[w] = self.classes[data[self.pos]]; self.pos += 1
        self.weights = {}
        for _ in range(rv()):
            f = rs(); n = data[self.pos]; self.pos += 1
            fw = {}
            for _ in range(n):
                t = self.classes[data[self.pos]]; self.pos += 1
                z = rv(); fw[t] = (z >> 1) if not z & 1 else -((z + 1) >> 1)
            self.weights[f] = fw

    def tag(self, toks):
        def score_int(f, scores):
            fw = self.weights.get(f)
            if fw:
                for tag, w in fw.items(): scores[tag] += w
        return predict_sentence(toks, self.tagdict, self.classes, score_int)

# ---------------------------------------------------------------- stage: fixtures
def stage_fixtures():
    model = BinModel()
    _, eval_ = load_tagged()
    stress = [["😀", "emoji", "🙂🙂", "test"], ["İstanbul", "STRASSE", "ǅungla"], ["a"], ["-"], [" "],
              ["\n\n"], ["ﬁle", "①", "²", "½"], ["ＡＢＣ", "ｗｉｄｅ"], ["don't", "y'all", "'tis"],
              ["123", "45.6", "7,890", "12:30", "$5", "£3.50"], ["snake_case", "camelCase", "kebab-case"]]
    with open(os.path.join(PARITY, 'en_tagger_fixtures.tsv'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('# held-out eval sentences + stress rows | json(tokens) \\t json(tags from en_tagger.bin.gz python reference)\n')
        for toks in [toks for _, toks, _ in eval_] + stress:
            f.write(f'{json.dumps(toks, ensure_ascii=False)}\t{json.dumps(model.tag(toks), ensure_ascii=False)}\n')
    print(f'fixtures: {len(eval_) + len(stress)} rows')

# ---------------------------------------------------------------- stage: parity
def stage_parity():
    model = BinModel()
    _, eval_ = load_tagged()
    acc, confusion = run_eval(eval_, model.tagdict, model.classes, lambda f, s: [s.__setitem__(t, s.get(t, 0) + w) for t, w in model.weights.get(f, {}).items()])
    total = sum(len(tags) for _, _, tags in eval_)
    print(f'held-out tag agreement vs spacy: {acc:.5f} ({total} tokens, {len(eval_)} sentences)')
    print('top confusions (spacy -> ours):', confusion.most_common(15))

    sys.path.insert(0, MISAKI)
    from misaki import en
    g2p = en.G2P(trf=False, british=False, fallback=lambda tk: (None, None), unk='❓')
    g2p.fallback = None
    real_nlp = g2p.nlp

    class DistilledNLP:
        def __call__(self, text):
            doc = real_nlp(text)
            for t, tag in zip(doc, model.tag([t.text for t in doc])):
                if t.tag_ != tag: t.tag_ = tag
            return doc

    reserved = []
    with open(os.path.join(PARITY, 'en_fixtures.tsv'), encoding='utf-8') as f:
        for line in f:
            if line.startswith('S\t'): reserved.append(unesc(line.rstrip('\n').split('\t')[3]))

    corpus_texts = {}
    for line in gzip.open(os.path.join(PARITY, 'en_tagger_corpus.tsv.gz'), 'rt', encoding='utf-8'):
        cat, s = line.rstrip('\n').split('\t')
        s = unesc(s)
        if is_eval(s): corpus_texts.setdefault(cat, []).append(s)
    rng = random.Random(777)
    eval_texts = [s for sents in corpus_texts.values() for s in sents]
    rng.shuffle(eval_texts)
    eval_texts = eval_texts[:6000]

    for name, texts in (('held-out eval', eval_texts), ('reserved en_fixtures', reserved)):
        same = diff = err = 0
        confused = Counter()
        for text in texts:
            try:
                g2p.nlp = real_nlp
                ref, ref_tokens = g2p(text)
                g2p.nlp = DistilledNLP()
                got, got_tokens = g2p(text)
            except Exception:
                err += 1; continue
            if ref == got: same += 1
            else:
                diff += 1
                for a, b in zip(ref_tokens, got_tokens):
                    if a.tag != b.tag: confused[(a.tag, b.tag)] += 1
        print(f'phoneme parity [{name}]: {same}/{same + diff} = {same / max(1, same + diff):.5f} ({err} oracle errors skipped)')
        print(f'  tag confusions on mismatched sentences (spacy -> ours): {confused.most_common(12)}')

STAGES = {'corpus': stage_corpus, 'tag': stage_tag, 'train': stage_train, 'export': stage_export,
          'fixtures': stage_fixtures, 'parity': stage_parity}
if __name__ == '__main__':
    for name in (sys.argv[1:] or ['all']):
        for s, fn in STAGES.items():
            if name in (s, 'all'): print(f'=== {s} ==='); fn()
