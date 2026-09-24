"""Probe: contextual calibration x option-marker choice.
Gemma 4 E4B 8-bit, 40 hard items (seed 1), S/Q/Q, 'Answer: **' / '{}'.
markers: letters | korean (random, non-sequential) | far (per item, embedding-farthest
from the prompt). calib: none | cc (Zhao 2021: state -> content-free, subtract log-prior)."""
import sys, copy, random, unicodedata
sys.path.insert(0, "src")
import mlx.core as mx
from ryotide.jevbench_adapter import MlxJevLocalAdapter
from jevbench.tasks import load_jsonl

tasks = random.Random(1).sample(load_jsonl("vendor/jevbench/datasets/public/hard.jsonl"), 40)
KW = dict(endpoint="mlx-community/gemma-4-e4b-it-8bit", orders=1, repeat=2,
          pin_prefix="Answer: **", pin_marker="{}")
ad = MlxJevLocalAdapter(**KW); clf = ad.load(); tok = clf.tokenizer
emb = clf.model.language_model.model.embed_tokens

def single(s):
    a = tok.encode(s, add_special_tokens=False); b = tok.encode(s + ".", add_special_tokens=False)
    return len(a) == 1 and b[:1] == a

# -- korean: random single-token syllables, excluding the ordinal ga-na-da sequence
ORDINAL = set("가나다라마바사아자차카타파하")
hangul = [chr(c) for c in range(0xAC00, 0xD7A4)]
KOREAN = random.Random(7).sample([h for h in hangul if single(h) and h not in ORDINAL], 16)

# -- far pool: printable single-char single tokens from several scripts, norm outliers dropped
def printable(ch):
    return unicodedata.category(ch)[0] in "LS" and not ch.isascii()
ranges = [(0xAC00, 0xD7A4), (0x4E00, 0x9FFF), (0x0370, 0x03FF), (0x0400, 0x04FF),
          (0x2190, 0x21FF), (0x2200, 0x22FF), (0x25A0, 0x25FF), (0x0E00, 0x0E7F), (0x0590, 0x05FF)]
pool = [chr(c) for a, b in ranges for c in range(a, b) if printable(chr(c)) and single(chr(c))]
pid = [tok.encode(p, add_special_tokens=False)[0] for p in pool]
E = emb(mx.array(pid)).astype(mx.float32)
nrm = mx.linalg.norm(E, axis=1); lo, hi = sorted(nrm.tolist())[len(pool)//10], sorted(nrm.tolist())[9*len(pool)//10]
keep = [i for i, v in enumerate(nrm.tolist()) if lo <= v <= hi]
pool = [pool[i] for i in keep]; E = E[mx.array(keep)]
E = E / mx.linalg.norm(E, axis=1, keepdims=True)
print(f"korean markers: {''.join(KOREAN)}   far pool: {len(pool)} chars after norm filter", flush=True)

def far_markers(t, k):
    probe = copy.copy(ad); probe.markers = list("ABCDEFGHIJKLMNOP")
    ids = sorted(set(tok.encode(probe._prompt(t, list(range(k))), add_special_tokens=False)))
    P = emb(mx.array(ids)).astype(mx.float32); P = P / mx.linalg.norm(P, axis=1, keepdims=True)
    sim = mx.max(E @ P.T, axis=1).tolist()          # closest prompt token, per candidate
    chosen = []
    for i in sorted(range(len(pool)), key=lambda i: sim[i]):
        if all(float((E[i] * E[j]).sum().item()) < 0.3 for j in chosen):
            chosen.append(i)
        if len(chosen) == k: break
    return [pool[i] for i in chosen], max(sim[i] for i in chosen)

def read(a, t, state_override=None):
    k = len(t.labels)
    tt = t if state_override is None else copy.copy(t)
    if state_override is not None: tt.state = state_override
    sel = mx.array(a._letters(k, "{}"))
    toks = a._tokens(a._prompt(tt, list(range(k))), suffix="Answer: **")
    if state_override is None:
        assert all(i in toks for i in a._letters(k, "{}")), "marker token not in prompt as-is"
    row = clf.logits_at_end(clf.prefill([]), toks)
    mass = float(mx.sum(mx.softmax(row.astype(mx.float32))[sel]).item())
    return mx.log(mx.softmax(row[sel].astype(mx.float32))), mass

ARMS = [f"{m}{c}" for m in ("letters", "korean", "far") for c in ("", "+cc")]
R = {a: [] for a in ARMS}; farsim = []
for n, t in enumerate(tasks, 1):
    k = len(t.labels); gold = t.labels.index(str(t.expected))
    for mname in ("letters", "korean", "far"):
        a = copy.copy(ad); a._letter_ids = {}
        a.markers = (list("ABCDEFGHIJKLMNOP") if mname == "letters" else
                     KOREAN if mname == "korean" else None)
        if mname == "far":
            a.markers, s = far_markers(t, k); farsim.append(s)
        lp, mass = read(a, t)
        prior = mx.mean(mx.stack([read(a, t, cf)[0] for cf in ("N/A", "", "[MASK]")]), axis=0)
        for arm, v in ((mname, lp), (mname + "+cc", lp - prior)):
            pred = int(mx.argmax(v).item())
            R[arm].append((pred == gold, mass, pred, gold, k))
    line = " ".join(f"{a}:{'ok' if R[a][-1][0] else 'x '}" for a in ARMS)
    print(f"[{n:2d}/40] {line}", flush=True)

print(f"\nfar markers: mean max-cos to prompt {sum(farsim)/len(farsim):.3f}", flush=True)
gpos = [0]*5
for r in R["letters"]: gpos[min(r[3],4)] += 1
print("gold position dist   " + " ".join(f"{x/40:.2f}" for x in gpos))
print(f"{'arm':12s} {'acc':>6s} {'mass mean':>10s} {'mass min':>9s}   predicted-position dist (A..E+)")
for a in ARMS:
    rs = R[a]; ms = [r[1] for r in rs]; pp = [0]*5
    for r in rs: pp[min(r[2],4)] += 1
    print(f"{a:12s} {sum(r[0] for r in rs)/40:6.3f} {sum(ms)/40:10.4f} {min(ms):9.4f}   "
          + " ".join(f"{x/40:.2f}" for x in pp))
