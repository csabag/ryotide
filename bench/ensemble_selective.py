"""Does averaging two models' distributions improve the ABSTENTION signal,
not just accuracy? Pure recalculation from saved runs."""
import json, sys
sys.path.insert(0, "src")
from ryotide.selective import curve, aurc, coverage_for_accuracy, at_coverage

load = lambda t: {r["task_id"]: r for r in (json.loads(l) for l in open(f"results/jevbench/{t}/results.jsonl"))}
A, B = load(sys.argv[1] if len(sys.argv) > 1 else "3.5-4b-8bit-cal2"), load(sys.argv[2] if len(sys.argv) > 2 else "gemma4-e4b-cal2")
ids = [i for i in A if i in B and A[i].get("probs") and B[i].get("probs")]
top = lambda p: max(p, key=p.get)

def build(name, fn):
    items = []
    for i in ids:
        p = fn(A[i]["probs"], B[i]["probs"])
        ok = (top(p) == top(A[i]["probs"])) and A[i]["correct"] or \
             (top(p) == top(B[i]["probs"])) and B[i]["correct"]
        # correctness of the ENSEMBLE prediction, resolved properly:
        pred = top(p)
        ok = (pred == top(A[i]["probs"]) and bool(A[i]["correct"])) or \
             (pred == top(B[i]["probs"]) and bool(B[i]["correct"]))
        items.append((max(p.values()), ok))
    pts = curve(items)
    return name, pts

STRATS = [
 ("Qwen3.5-4B alone", lambda a,b: a),
 ("Gemma E4B alone",  lambda a,b: b),
 ("mean of distributions", lambda a,b: {k:(a[k]+b[k])/2 for k in a}),
 ("geometric mean", lambda a,b: (lambda d:{k:v/sum(d.values()) for k,v in d.items()})({k:(a[k]*b[k])**0.5 for k in a})),
]
print(f"{'strategy':24s} {'AURC':>6s} {'all':>6s} {'@90%':>6s} {'@70%':>6s} {'@50%':>6s}  cov@85 cov@90 cov@95")
print("-"*82)
for name, fn in STRATS:
    _, pts = build(name, fn)
    cv = "  ".join(f"{(coverage_for_accuracy(pts,t).coverage if coverage_for_accuracy(pts,t) else 0)*100:5.0f}%" for t in (0.85,0.90,0.95))
    print(f"{name:24s} {aurc(pts):6.3f} {pts[-1].selective_accuracy:6.3f} "
          f"{at_coverage(pts,.9).selective_accuracy:6.3f} {at_coverage(pts,.7).selective_accuracy:6.3f} "
          f"{at_coverage(pts,.5).selective_accuracy:6.3f}  {cv}")
