"""Sweep candidate answer prefixes on a sample: marker mass AND accuracy.

Marker mass alone only says the read landed on a marker; it says nothing about
whether the resulting distribution is any good. Both are measured here.
"""
import json, random, sys
sys.path.insert(0, "src")
import mlx.core as mx
from ryotide.jevbench_adapter import MlxJevLocalAdapter
from jevbench.tasks import load_jsonl

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3.5-4B-MLX-8bit"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
tasks = []
for f in ("original", "easy", "hard"):
    tasks += load_jsonl(f"vendor/jevbench/datasets/public/{f}.jsonl")
sample = random.Random(1).sample(tasks, N)

CANDIDATES = [
    ("english  Answer: **",      "Answer: **"),
    ("english  Answer:",         "Answer:"),
    ("english  The answer is",   "The answer is"),
    ("chinese  答案：**",  "答案：**"),      # 答案：**
    ("chinese  答案：",    "答案："),        # 答案：
    ("chinese  答：",          "答："),              # 答：
    ("chinese  选择：**",  "选择：**"),      # 选择：**
    ("chinese  回答：**",  "回答：**"),      # 回答：**
]
ad = MlxJevLocalAdapter(endpoint=MODEL, orders=1); clf = ad.load()
print(f"model={MODEL}   n={N} tasks\n")
print(f"{'prefix':26s} {'marker mass':>12s} {'accuracy':>9s} {'mean conf':>10s} {'tok':>5s}")
print("-" * 68)
for label, suf in CANDIDATES:
    masses, ok, confs, extra = [], 0, [], None
    for t in sample:
        k = len(t.labels)
        for pat in ("{}", " {}"):
            try:
                sel = mx.array(ad._letters(k, pat))
            except ValueError:
                continue
            toks = ad._tokens(ad._prompt(t, list(range(k))), suffix=suf)
            row = clf.logits_at_end(clf.prefill([]), toks)
            full = mx.softmax(row.astype(mx.float32))
            m = float(mx.sum(full[sel]).item())
            if pat == "{}" or m > masses[-1] if masses else True:
                pass
            break   # use the first resolvable pattern, consistent across prefixes
        p = mx.softmax(row[sel].astype(mx.float32)).tolist()
        i = max(range(k), key=lambda j: p[j])
        masses.append(m); confs.append(p[i]); ok += (t.labels[i] == str(t.expected))
        if extra is None:
            extra = len(clf.tokenizer.encode(suf, add_special_tokens=False))
    print(f"{label:26s} {sum(masses)/len(masses):12.4f} {ok/len(sample):9.3f} "
          f"{sum(confs)/len(confs):10.3f} {extra:5d}")
