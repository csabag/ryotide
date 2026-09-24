"""Probe: interleave '(question will be: <q>)' after every state sentence.
Gemma 4 E4B 8-bit, 40 hard items, best config otherwise. Both arms in-process."""
import sys, re, random, time
sys.path.insert(0, "src")
import mlx.core as mx
from ryotide.jevbench_adapter import MlxJevLocalAdapter, _state_text
from jevbench.tasks import load_jsonl

SPLIT = re.compile(r"((?<=[.!?])[ \t]+|\n+)")   # keep separators so layout survives

def interleave(state, q):
    parts, out = SPLIT.split(state), []
    for p in parts:
        out.append(p)
        if not SPLIT.fullmatch(p or " ") and re.search(r"[A-Za-z]", p):
            out.append(f" (question will be: {q})")
    return "".join(out)

class Interleaved(MlxJevLocalAdapter):
    def _prompt(self, task, order):
        full, state = super()._prompt(task, order), _state_text(task.state)
        assert full.startswith(state)
        return interleave(state, task.question["instructions"]) + full[len(state):]

tasks = random.Random(1).sample(load_jsonl("vendor/jevbench/datasets/public/hard.jsonl"), 40)
kw = dict(endpoint="mlx-community/gemma-4-e4b-it-8bit", orders=1, repeat=2,
          pin_prefix="Answer: **", pin_marker="{}")
base = MlxJevLocalAdapter(**kw); clf = base.load()
inter = Interleaved(**kw); inter._clf = clf; inter._letter_ids = base._letter_ids

def score(ad, t):
    k = len(t.labels); sel = mx.array(ad._letters(k, "{}"))
    toks = ad._tokens(ad._prompt(t, list(range(k))), suffix="Answer: **")
    t0 = time.perf_counter()
    row = clf.logits_at_end(clf.prefill([]), toks)
    mass = float(mx.sum(mx.softmax(row.astype(mx.float32))[sel]).item())
    p = mx.softmax(row[sel].astype(mx.float32)).tolist()
    i = max(range(k), key=lambda j: p[j])
    return t.labels[i] == str(t.expected), mass, p[i], len(toks), time.perf_counter() - t0

print("sample prompt head:\n" + inter._prompt(tasks[0], list(range(len(tasks[0].labels))))[:500] + "\n...", flush=True)
R = {"base": [], "inter": []}
for n, t in enumerate(tasks, 1):
    for name, ad in (("base", base), ("inter", inter)):
        R[name].append(score(ad, t))
    b, i = R["base"][-1], R["inter"][-1]
    flag = "" if b[0] == i[0] else ("  <- inter FIXED" if i[0] else "  <- inter BROKE")
    print(f"[{n:2d}/40] base {'ok ' if b[0] else 'x  '} inter {'ok ' if i[0] else 'x  '}"
          f" tok {b[3]:5d}->{i[3]:5d}  run acc base {sum(r[0] for r in R['base'])/n:.3f}"
          f" inter {sum(r[0] for r in R['inter'])/n:.3f}{flag}", flush=True)

print("\narm     acc    mass_mean mass_min  conf   tokens  sec/item")
for name, rs in R.items():
    m = [r[1] for r in rs]
    print(f"{name:6s} {sum(r[0] for r in rs)/len(rs):.3f}  {sum(m)/len(m):.4f}   {min(m):.4f}  "
          f"{sum(r[2] for r in rs)/len(rs):.3f}  {sum(r[3] for r in rs)//len(rs):6d}  {sum(r[4] for r in rs)/len(rs):.2f}")
