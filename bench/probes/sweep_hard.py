"""Stage 1: (prefix x marker-form) sweep on the HARD tier -- mass AND accuracy."""
import sys, random
sys.path.insert(0, "src")
import mlx.core as mx
from ryotide.jevbench_adapter import MlxJevLocalAdapter
from jevbench.tasks import load_jsonl

MODEL = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
tasks = load_jsonl("vendor/jevbench/datasets/public/hard.jsonl")
sample = random.Random(1).sample(tasks, min(N, len(tasks)))

ad = MlxJevLocalAdapter(endpoint=MODEL, orders=1, repeat=2)
clf = ad.load()
print(f"model={MODEL}  hard tier, n={len(sample)}", flush=True)
print(f"{'prefix':40s} {'pat':6s} {'mass':>8s} {'acc':>7s} {'conf':>7s} {'tok':>4s}", flush=True)
print("-" * 78, flush=True)
rows = []
for pat in ad.PATTERNS:
    for suf in ad.SUFFIXES:
        masses, confs, ok, n = [], [], 0, 0
        for t in sample:
            k = len(t.labels)
            try:
                sel = mx.array(ad._letters(k, pat))
            except Exception:
                continue
            toks = ad._tokens(ad._prompt(t, list(range(k))), suffix=suf)
            row = clf.logits_at_end(clf.prefill([]), toks)
            masses.append(float(mx.sum(mx.softmax(row.astype(mx.float32))[sel]).item()))
            p = mx.softmax(row[sel].astype(mx.float32)).tolist()
            i = max(range(k), key=lambda j: p[j])
            confs.append(p[i]); ok += (t.labels[i] == str(t.expected)); n += 1
        if not n:
            continue
        ntok = len(clf.tokenizer.encode(suf, add_special_tokens=False))
        rows.append((sum(masses)/n, ok/n, sum(confs)/n, suf, pat, ntok))
        print(f"{repr(suf)[:40]:40s} {repr(pat):6s} {rows[-1][0]:8.4f} "
              f"{rows[-1][1]:7.3f} {rows[-1][2]:7.3f} {ntok:4d}", flush=True)
print("\nbest by mass:    ", max(rows)[3:5], flush=True)
print("best by accuracy:", max(rows, key=lambda r: r[1])[3:5], flush=True)
