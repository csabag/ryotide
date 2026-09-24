"""Probe: confidence-gated reasoning. Gemma 4 E4B 8-bit, 40 hard items (seed 1), S/Q/Q.
Every item gets BOTH the fast one-pass read and a reasoning pass, so the gate threshold
can be swept offline. The reasoning pass still ends in the same masked marker read."""
import sys, json, random, time
sys.path.insert(0, "src")
import mlx.core as mx
from mlx_lm import stream_generate
from ryotide.jevbench_adapter import MlxJevLocalAdapter
from jevbench.tasks import load_jsonl

REASON = "Let me think step by step.\n"
MAX_GEN = 384
tasks = random.Random(1).sample(load_jsonl("vendor/jevbench/datasets/public/hard.jsonl"), 40)
ad = MlxJevLocalAdapter(endpoint="mlx-community/gemma-4-e4b-it-8bit", orders=1, repeat=2,
                        pin_prefix="Answer: **", pin_marker="{}")
clf = ad.load()

def read(toks, k):
    sel = mx.array(ad._letters(k, "{}"))
    row = clf.logits_at_end(clf.prefill([]), toks)
    mass = float(mx.sum(mx.softmax(row.astype(mx.float32))[sel]).item())
    p = mx.softmax(row[sel].astype(mx.float32)).tolist()
    i = max(range(k), key=lambda j: p[j])
    return i, p[i], mass

out = open("results/probes/gated_items.jsonl", "w")
for n, t in enumerate(tasks, 1):
    k = len(t.labels); gold = t.labels.index(str(t.expected))
    prompt = ad._prompt(t, list(range(k)))
    t0 = time.perf_counter()
    fi, fc, fm = read(ad._tokens(prompt, suffix="Answer: **"), k)
    tf = time.perf_counter() - t0

    t0 = time.perf_counter(); text, ngen = "", 0
    for r in stream_generate(clf.model, clf.tokenizer, ad._tokens(prompt, suffix=REASON),
                             max_tokens=MAX_GEN):
        text += r.text; ngen += 1
        if "Answer" in text[-40:] or "answer is" in text[-40:]:
            break
    for stop in ("**Answer", "Answer:", "Final answer", "The answer is", "answer is"):
        if stop in text:
            text = text[:text.index(stop)]
    ri, rc, rm = read(ad._tokens(prompt, suffix=REASON + text.rstrip() + "\n\nAnswer: **"), k)
    tr = time.perf_counter() - t0

    rec = dict(id=t.task_id if hasattr(t, "task_id") else t.id, family=t.family if hasattr(t, "family") else None,
               k=k, gold=gold, fast_pred=fi, fast_conf=fc, fast_mass=fm, fast_s=tf,
               rsn_pred=ri, rsn_conf=rc, rsn_mass=rm, rsn_s=tr, gen_tokens=ngen, rationale=text)
    out.write(json.dumps(rec) + "\n"); out.flush()
    tag = ("" if (fi == gold) == (ri == gold) else "  <- reasoning FIXED" if ri == gold else "  <- reasoning BROKE")
    print(f"[{n:2d}/40] fast {'ok' if fi==gold else 'x '} conf {fc:.3f} | reason {'ok' if ri==gold else 'x '}"
          f" conf {rc:.3f} mass {rm:.3f} gen {ngen:3d} tok {tr:5.1f}s{tag}", flush=True)
    if n == 1:
        print("--- first rationale ---\n" + text[:700] + "\n---", flush=True)
