"""Risk-coverage curves from saved JevBench runs. No re-run needed."""
import json, sys
sys.path.insert(0, "src")
from ryotide.selective import curve, aurc, at_coverage, coverage_for_accuracy, score_of

TAGS = sys.argv[1:] or ["3.5-4b-8bit-o1","3.5-4b-o1","3.5-2b-o1","1.5b-o4",
                        "gemma4-e2b-8bit-o1","0.5b-8bit-base","0.5b-coder-8bit"]
out = {}
for tag in TAGS:
    try:
        rs = [json.loads(l) for l in open(f"results/jevbench/{tag}/results.jsonl")]
    except FileNotFoundError:
        continue
    for how in ("confidence", "margin"):
        items = [(score_of(r["probs"], how), bool(r["correct"])) for r in rs if r.get("probs")]
        pts = curve(items)
        full = pts[-1]
        row = {"aurc": round(aurc(pts), 4), "full_accuracy": round(full.selective_accuracy, 4)}
        for c in (0.9, 0.8, 0.7, 0.5):
            p = at_coverage(pts, c)
            row[f"acc@{int(c*100)}"] = round(p.selective_accuracy, 4)
        for t in (0.85, 0.90, 0.95):
            p = coverage_for_accuracy(pts, t)
            row[f"cov@{int(t*100)}"] = round(p.coverage, 4) if p else None
        out.setdefault(tag, {})[how] = row

print(f"{'run':22s} {'score':11s} {'AURC':>6s} {'100%':>6s} {'90%':>6s} {'80%':>6s} {'70%':>6s} {'50%':>6s}   cov@85 cov@90 cov@95")
print("-"*104)
for tag, d in out.items():
    for how, r in d.items():
        cv = "  ".join(f"{(r[f'cov@{t}'] if r[f'cov@{t}'] is not None else 0)*100:5.0f}%" for t in (85,90,95))
        print(f"{tag:22s} {how:11s} {r['aurc']:6.3f} {r['full_accuracy']:6.3f} {r['acc@90']:6.3f} "
              f"{r['acc@80']:6.3f} {r['acc@70']:6.3f} {r['acc@50']:6.3f}   {cv}")
json.dump(out, open("results/jevbench/selective.json","w"), indent=1)
print("\nAURC = area under risk-coverage curve, lower is better. cov@N = most decisions answerable at N% accuracy.")
