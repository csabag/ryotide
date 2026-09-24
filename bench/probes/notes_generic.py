"""S / Notes / Q / Q probe, parameterised. Notes are generated on discarded KV branches
over 128-token chunks of the clean state (word-boundary snapped), then listed once in
'Mental notes: ...' after the state; same masked read as baseline. 40 hard items, seed 1.
usage: notes_generic.py MODEL LANG(en|zh) FRAME TAG [words3]   FRAME may contain {q}.
words3: English budget is 3 whitespace-separated words (max 10 tokens) instead of 3 tokens."""
import sys, json, random, time
sys.path.insert(0, "src")
import mlx.core as mx
from ryotide.jevbench_adapter import MlxJevLocalAdapter, _state_text
from ryotide.branch import fork_cache
from jevbench.tasks import load_jsonl

MODEL, LANG, FRAME, TAG = sys.argv[1:5]
OPTS = set(sys.argv[5:])
WORDS3 = "words3" in OPTS          # English budget: 3 whitespace words (max 10 tokens)
NOFILLER = "nofiller" in OPTS      # English: mask scaffolding/format tokens during note decode
MIN_TOK, CHUNK_TOK, BUDGET = 128, 128, 3           # en: tokens; zh: characters
HEAD = {"gemma": "<bos><|turn>user\n", "qwen": "<|im_start|>user\n"}["gemma" if "gemma" in MODEL else "qwen"]
tasks = random.Random(1).sample(load_jsonl("vendor/jevbench/datasets/public/hard.jsonl"), 40)
ad = MlxJevLocalAdapter(endpoint=MODEL, orders=1, repeat=2, pin_prefix="Answer: **", pin_marker="{}")
clf = ad.load(); tok = clf.tokenizer
enc = lambda s: tok.encode(s, add_special_tokens=False)
print(f"model={MODEL} lang={LANG} frame={FRAME!r} budget={'3 words' if WORDS3 else BUDGET} nofiller={NOFILLER}", flush=True)

FILLER = set("""the a an this that these those it its is are was were be been being to of for and or
check ensure verify need needs please note i we you will would should must can could if whether
key fact facts important words word question relevant most here there so then as also""".split())
FORMAT = set('*"\'`:#([>')
if LANG == "en" and NOFILLER:
    NV = len(tok.get_vocab()); _txt = [tok.decode([i]) for i in range(NV)]
    def _banned(s):
        w = s.strip()
        return (not w) or all(c in FORMAT for c in w) or w.lower() in FILLER
    stops = {i for i, s in enumerate(_txt) if ")" in s or "\n" in s}
    ban = mx.zeros((NV,)); ban[mx.array([i for i, s in enumerate(_txt) if _banned(s) and i not in stops])] = -1e9
    print(f"filler ban: {int((ban < 0).sum().item())} tokens masked", flush=True)

if LANG == "zh":
    CLOSE = "）"; assert len(enc(CLOSE)) == 1; CLOSE_ID = enc(CLOSE)[0]
    NV = len(tok.get_vocab()); text_of = [tok.decode([i]) for i in range(NV)]
    banned = set(c for c in FRAME if 0x4E00 <= ord(c) <= 0x9FFF)   # no echoing the frame
    ok = lambda s: s and all(0x4E00 <= ord(c) <= 0x9FFF for c in s) and not (set(s) & banned)
    allow = mx.full((NV,), -1e9); allow[mx.array([i for i, s in enumerate(text_of) if ok(s)] + [CLOSE_ID])] = 0.0
    print(f"CJK note tokens allowed: {int((allow == 0).sum().item()) - 1}, banned frame chars: {''.join(sorted(banned))}", flush=True)

# -- gate: branch rollback must be bit-exact on this model
_st = max((_state_text(t.state) for t in tasks), key=len)
_A, _B = enc(HEAD + _st[:4000]), enc(_st[4000:4600])
_ref = clf.logits_at_end(clf.prefill(_A), _B); _c = clf.prefill(_A)
for _ in range(3):
    _br = fork_cache(_c); _r = clf.logits_at_end(_br, enc(" (note"))
    for _ in range(3): _r = clf.logits_at_end(_br, [int(mx.argmax(_r).item())])
    del _br
_d = float(mx.max(mx.abs(_ref.astype(mx.float32) - clf.logits_at_end(_c, _B).astype(mx.float32))).item())
print(f"rollback identity control: maxdiff {_d}", flush=True); assert _d == 0.0

def chunks(ids):
    out, i = [], 0
    while i < len(ids):
        j = min(len(ids), i + CHUNK_TOK); k = j
        while k < len(ids) and k - j < 16 and not tok.decode([ids[k]])[:1].isspace(): k += 1
        j = k if (k < len(ids) and tok.decode([ids[k]])[:1].isspace()) else j
        out.append(ids[i:j]); i = j
    return out

def note_on(br, row):
    if LANG == "en":
        ids = []
        for _ in range(10 if WORDS3 else BUDGET):
            r = (row[:NV].astype(mx.float32) + ban) if NOFILLER else row
            t = int(mx.argmax(r).item()); piece = tok.decode([t])
            if ")" in piece or "\n" in piece or t == tok.eos_token_id: break
            if WORDS3 and len(tok.decode(ids + [t]).split()) > 3: break   # a 4th word starts
            ids.append(t); row = clf.logits_at_end(br, [t])
        return tok.decode(ids).strip()
    note = ""
    while len(note) < BUDGET:
        t = int(mx.argmax(row[:NV].astype(mx.float32) + allow).item())
        if t == CLOSE_ID: break
        piece = text_of[t]
        if len(note) + len(piece) > BUDGET: note += piece[: BUDGET - len(note)]; break
        note += piece; row = clf.logits_at_end(br, [t])
    return note

def take_notes(state, q):
    cache = clf.prefill(enc(HEAD)); notes = []; frame = FRAME.replace("{q}", q)
    for c in chunks(enc(state)):
        clf.logits_at_end(cache, c)
        br = fork_cache(cache); notes.append(note_on(br, clf.logits_at_end(br, enc(frame)))); del br
    del cache
    return [n for n in notes if n]

def read(prompt, k):
    sel = mx.array(ad._letters(k, "{}"))
    row = clf.logits_at_end(clf.prefill([]), ad._tokens(prompt, suffix="Answer: **"))
    mass = float(mx.sum(mx.softmax(row.astype(mx.float32))[sel]).item())
    p = mx.softmax(row[sel].astype(mx.float32)).tolist(); i = max(range(k), key=lambda j: p[j])
    return i, p[i], mass

out = open(f"results/probes/notes_{TAG}_items.jsonl", "w"); R = {"base": [], "notes": []}; shown = False
for n, t in enumerate(tasks, 1):
    k = len(t.labels); gold = t.labels.index(str(t.expected))
    full, state = ad._prompt(t, list(range(k))), _state_text(t.state); assert full.startswith(state)
    t0 = time.perf_counter(); b = read(full, k); tb = time.perf_counter() - t0
    t0 = time.perf_counter()
    notes = take_notes(state, t.question["instructions"]) if len(enc(state)) > MIN_TOK else []
    np_ = (state + "\n\nMental notes: " + ", ".join(notes) + full[len(state):]) if notes else full
    m = read(np_, k); tn = time.perf_counter() - t0
    R["base"].append((b[0] == gold, b[2], tb)); R["notes"].append((m[0] == gold, m[2], tn))
    out.write(json.dumps(dict(n=n, k=k, gold=gold, notes=notes, base=b, notes_read=m), ensure_ascii=False) + "\n"); out.flush()
    tag = "" if (b[0]==gold) == (m[0]==gold) else ("  <- notes FIXED" if m[0]==gold else "  <- notes BROKE")
    print(f"[{n:2d}/40] tok {len(enc(state)):5d} notes {len(notes):2d} | base {'ok' if b[0]==gold else 'x '}"
          f" notes {'ok' if m[0]==gold else 'x '} mass {m[2]:.3f} {tn:5.1f}s{tag}", flush=True)
    if notes and not shown:
        print("--- first notes --- " + " | ".join(notes[:20]), flush=True); shown = True

allN = [x for l in open(f"results/probes/notes_{TAG}_items.jsonl") for x in json.loads(l)["notes"]]
print(f"\nnotes {len(allN)}, distinct {len(set(allN))}")
print("arm     acc    mass_mean mass_min  sec/item")
for a, rs in R.items():
    ms = [r[1] for r in rs]
    print(f"{a:6s} {sum(r[0] for r in rs)/40:.3f}  {sum(ms)/40:.4f}   {min(ms):.4f}  {sum(r[2] for r in rs)/40:.2f}")
