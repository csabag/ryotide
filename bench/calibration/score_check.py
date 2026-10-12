"""Ordinal ("score") questions: where the engine goes wrong, on a served run.

Per item: gold level, argmax level, expected level (sum k * p_k). Reports exact accuracy,
chance-corrected accuracy (chance = 1 / levels), within-one, mean signed error (positive =
rates too high), expected-value accuracy (round(EV) == gold), and the confusion counts.

    python3 bench/calibration/score_check.py RUN/results.jsonl TASKS.jsonl
"""
import json, sys
from collections import Counter

res = {json.loads(l)["task_id"]: json.loads(l) for l in open(sys.argv[1])}
tasks = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[2])}
by = {}
for tid, t in tasks.items():
    if t["question"].get("type") != "score" or tid not in res:
        continue
    r = res[tid]; p = r["probs"]; labels = t["labels"]
    probs = [p.get(l, 0.0) for l in labels]
    am = max(range(len(labels)), key=probs.__getitem__)
    ev = sum(k * x for k, x in enumerate(probs)) / (sum(probs) or 1)
    kind = tid.rsplit("-", 1)[1]
    by.setdefault(kind, []).append((t["expected"], am, ev, len(labels), max(probs)))
for kind, xs in sorted(by.items()):
    n = len(xs); L = xs[0][3]
    acc = sum(g == a for g, a, *_ in xs) / n
    print(f"== {kind}: n={n}, levels={L}")
    print(f"   exact {acc:.1%}  cc {100 * (acc - 1 / L) / (1 - 1 / L):.1f}  within-one "
          f"{sum(abs(g - a) <= 1 for g, a, *_ in xs) / n:.1%}  EV-rounded {sum(round(ev) == g for g, _, ev, *_ in xs) / n:.1%}"
          f"  mean signed err {sum(a - g for g, a, *_ in xs) / n:+.2f}  mean top-p {sum(x[4] for x in xs) / n:.2f}")
    print("   gold -> argmax:", dict(sorted(Counter((g, a) for g, a, *_ in xs).items())))
