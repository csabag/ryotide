"""Probe: S / Notes / Q / Q with CHINESE notes. For states > 500 chars, read the state in
~250-char chunks; after each, '（备忘：' + up to 3 hanzi (decode masked to CJK-only tokens)
+ '）'. Question-blind. Final prompt = clean state + 'Mental notes: a, b, ...' + Q/Q,
same masked read. Qwen3.5-4B 8-bit (hybrid: linear-attention + full attention), 40 hard items (seed 1). Baseline arm in-process."""
import sys, json, random, time
sys.path.insert(0, "src")
import mlx.core as mx
from ryotide.jevbench_adapter import MlxJevLocalAdapter, _state_text
from ryotide.branch import fork_cache
from jevbench.tasks import load_jsonl

MIN_TOK, CHUNK_TOK, NOTE_CHARS = 128, 128, 3
PREFIX, CLOSE = "（备忘：", "）"
HEAD = "<|im_start|>user\n"          # Qwen template up to the user content
tasks = random.Random(1).sample(load_jsonl("vendor/jevbench/datasets/public/hard.jsonl"), 40)
ad = MlxJevLocalAdapter(endpoint="mlx-community/Qwen3.5-4B-MLX-8bit", orders=1, repeat=2,
                        pin_prefix="Answer: **", pin_marker="{}")
clf = ad.load(); tok = clf.tokenizer
enc = lambda s: tok.encode(s, add_special_tokens=False)
cjk = lambda ch: 0x4E00 <= ord(ch) <= 0x9FFF

NV = len(tok.get_vocab())
text_of = [tok.decode([i]) for i in range(NV)]
CJK_IDS = [i for i, s in enumerate(text_of) if s and all(cjk(c) for c in s)]
CLOSE_ID = enc(CLOSE)[0]
assert len(enc(CLOSE)) == 1
allow = mx.full((NV,), -1e9); allow[mx.array(CJK_IDS + [CLOSE_ID])] = 0.0
print(f"vocab {NV}, CJK-only tokens allowed in notes: {len(CJK_IDS)}", flush=True)

# -- gate: rollback must be bit-exact on this hybrid architecture, or we don't run
from ryotide.branch import fork_cache as _fork
_st = max((_state_text(t.state) for t in tasks), key=len)
_A, _B = enc(HEAD + _st[:4000]), enc(_st[4000:4600])
_ref = clf.logits_at_end(clf.prefill(_A), _B)
_c = clf.prefill(_A)
for _ in range(3):
    _br = _fork(_c); _r = clf.logits_at_end(_br, enc(PREFIX))
    for _ in range(3):
        _r = clf.logits_at_end(_br, [int(mx.argmax(_r).item())])
    del _br
_got = clf.logits_at_end(_c, _B)
_d = float(mx.max(mx.abs(_ref.astype(mx.float32) - _got.astype(mx.float32))).item())
print(f"rollback identity control on {len(_A)}-token prefix: maxdiff {_d}", flush=True)
assert _d == 0.0, "KV/recurrent-state rollback is NOT clean on this model -- refusing to run"

def chunks(ids):
    """~CHUNK_TOK-token chunks of the state's own tokenization. Each cut snaps
    forward (at most 16 tokens) to a token that starts a word or line, so a note
    never lands mid-word."""
    out, i = [], 0
    while i < len(ids):
        j = min(len(ids), i + CHUNK_TOK)
        k = j
        while k < len(ids) and k - j < 16 and not tok.decode([ids[k]])[:1].isspace():
            k += 1
        j = k if (k < len(ids) and tok.decode([ids[k]])[:1].isspace()) else j
        out.append(ids[i:j]); i = j
    return out

def take_notes(state):
    """Notes see only the CLEAN state read so far (forked branch, then discarded)."""
    cache = clf.prefill(enc(HEAD)); notes = []
    for c in chunks(enc(state)):
        clf.logits_at_end(cache, c)
        br = fork_cache(cache)
        row = clf.logits_at_end(br, enc(PREFIX))
        note = ""
        while len(note) < NOTE_CHARS:
            t = int(mx.argmax(row[:NV].astype(mx.float32) + allow).item())
            if t == CLOSE_ID:
                break
            piece = text_of[t]
            if len(note) + len(piece) > NOTE_CHARS:
                note += piece[: NOTE_CHARS - len(note)]; break
            note += piece; row = clf.logits_at_end(br, [t])
        del br
        notes.append(note)
    del cache
    return [n for n in notes if n]

def read(prompt, k):
    sel = mx.array(ad._letters(k, "{}"))
    row = clf.logits_at_end(clf.prefill([]), ad._tokens(prompt, suffix="Answer: **"))
    mass = float(mx.sum(mx.softmax(row.astype(mx.float32))[sel]).item())
    p = mx.softmax(row[sel].astype(mx.float32)).tolist()
    i = max(range(k), key=lambda j: p[j])
    return i, p[i], mass

out = open("results/probes/notes_zh_qwen_items.jsonl", "w"); R = {"base": [], "notes": []}; nstates = 0
for n, t in enumerate(tasks, 1):
    k = len(t.labels); gold = t.labels.index(str(t.expected))
    full, state = ad._prompt(t, list(range(k))), _state_text(t.state)
    assert full.startswith(state)
    t0 = time.perf_counter(); b = read(full, k); tb = time.perf_counter() - t0
    t0 = time.perf_counter()
    notes = take_notes(state) if len(enc(state)) > MIN_TOK else []
    nstates += bool(notes)
    np_ = (state + "\n\nMental notes: " + ", ".join(notes) + full[len(state):]) if notes else full
    m = read(np_, k); tn = time.perf_counter() - t0
    R["base"].append((b[0] == gold, b[2], tb)); R["notes"].append((m[0] == gold, m[2], tn))
    out.write(json.dumps(dict(n=n, k=k, gold=gold, state_chars=len(state), notes=notes,
                              base=b, notes_read=m, base_s=tb, notes_s=tn), ensure_ascii=False) + "\n"); out.flush()
    tag = "" if (b[0]==gold) == (m[0]==gold) else ("  <- notes FIXED" if m[0]==gold else "  <- notes BROKE")
    print(f"[{n:2d}/40] tok {len(enc(state)):5d} notes {len(notes):2d} | base {'ok' if b[0]==gold else 'x '}"
          f" notes {'ok' if m[0]==gold else 'x '} mass {m[2]:.3f} {tn:5.1f}s{tag}", flush=True)
    if notes and nstates == 1:
        print("--- first notes --- " + " | ".join(notes[:25]), flush=True)

print(f"\nstates that got notes: {nstates}/40")
print("arm     acc    mass_mean mass_min  sec/item")
for a, rs in R.items():
    ms = [r[1] for r in rs]
    print(f"{a:6s} {sum(r[0] for r in rs)/40:.3f}  {sum(ms)/40:.4f}   {min(ms):.4f}  {sum(r[2] for r in rs)/40:.2f}")
