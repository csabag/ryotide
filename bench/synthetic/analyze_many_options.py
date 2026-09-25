"""Per-menu-size report for the many-option runs.

    uv run python bench/synthetic/analyze_many_options.py many-gemma-default [many-gemma-forced]
"""
import json
import sys
from collections import defaultdict

runs = {t: {r["task_id"]: r for r in map(json.loads, open(f"results/jevbench/{t}/results.jsonl"))}
        for t in sys.argv[1:]}
size = {}
for line in open("data/synthetic/many-options-v1.jsonl"):
    d = json.loads(line)
    size[d["id"]] = len(d["labels"])

print(f"{'run':22s} {'k':>4s} {'reader':>8s} {'acc':>6s} {'chance':>7s} {'mass p5':>8s} {'mass<0.9':>9s} {'p50 ms':>7s}")
for tag, R in runs.items():
    by = defaultdict(list)
    for tid, r in R.items():
        by[size[tid]].append(r)
    for k in sorted(by):
        rs = by[k]
        ok = [r for r in rs if r["ok"]]
        mass = sorted(r["runtime"]["marker_mass"] for r in ok)
        lat = sorted(r["latency_s"] for r in ok)
        reader = "codes" if "codes" in (ok[0]["runtime"].get("markers") or "") else "letters"
        print(f"{tag:22s} {k:4d} {reader:>8s} {sum(r['correct'] for r in rs)/len(rs):6.3f} {1/k:7.3f} "
              f"{mass[len(mass)//20]:8.3f} {sum(m < 0.9 for m in mass):9d} {lat[len(lat)//2]*1000:7.0f}"
              + ("" if len(ok) == len(rs) else f"   ({len(rs)-len(ok)} failed)"))
if len(runs) == 2:
    (a, A), (b, B) = runs.items()
    print("\nsame items, letters vs forced codes (k <= 26):")
    for k in sorted({size[t] for t in A if size[t] <= 26}):
        ids = [t for t in A if size[t] == k]
        same = sum(A[t]["predicted"] == B[t]["predicted"] for t in ids)
        print(f"  k={k}: {a} {sum(A[t]['correct'] for t in ids)}/{len(ids)}  {b} {sum(B[t]['correct'] for t in ids)}/{len(ids)}"
              f"  identical decisions {same}/{len(ids)}")
