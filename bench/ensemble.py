"""Combine two saved runs per-decision and rescore with the harness's scorer.

Every strategy is computed from probabilities already on disk -- no model runs.
Latency is summed per decision, because an ensemble pays for both models.
"""
import json, sys, collections
sys.path.insert(0, "src")
from jevbench import composite_v13 as C
from jevbench.metrics import ece_top_label, percentile
from jevbench.tasks import load_jsonl

A_TAG, B_TAG = sys.argv[1], sys.argv[2]
tasks = {}
for f in ("original", "easy", "hard"):
    for t in load_jsonl(f"vendor/jevbench/datasets/public/{f}.jsonl"):
        tasks[t.id] = t
TIER = {"original": "standard", "easy": "easy", "hard": "hard"}
def tier(tid):
    return "easy" if tid.startswith("easy-") else "standard" if tid.startswith("original-") else "hard"

def load(tag):
    return {r["task_id"]: r for r in (json.loads(l) for l in open(f"results/jevbench/{tag}/results.jsonl"))}
A, B = load(A_TAG), load(B_TAG)
ids = [i for i in A if i in B and A[i].get("probs") and B[i].get("probs")]

def top(p): return max(p, key=p.get)
def conf(p): return max(p.values())

# Raw confidences are not comparable across models with different calibration:
# E2B's 0.99 means ~44% on hard items, E4B's means far more. Rank each model's
# confidence inside its OWN distribution so the comparison is like-for-like.
def ranks(M):
    vals = sorted(conf(M[i]["probs"]) for i in ids)
    import bisect
    return lambda p: bisect.bisect_left(vals, conf(p)) / max(len(vals) - 1, 1)
RA, RB = ranks(A), ranks(B)

# Temperature that makes a model's mean confidence match its own accuracy.
def tempered(p, T):
    import math
    ks = list(p); xs = [math.log(max(p[k], 1e-9)) / T for k in ks]
    m = max(xs); ex = [math.exp(x - m) for x in xs]; s = sum(ex)
    return {k: e / s for k, e in zip(ks, ex)}

STRATS = {
 "A alone": lambda a,b: a["probs"],
 "B alone": lambda a,b: b["probs"],
 "max-confidence wins": lambda a,b: a["probs"] if conf(a["probs"]) >= conf(b["probs"]) else b["probs"],
 "mean of distributions": lambda a,b: {k:(a["probs"][k]+b["probs"][k])/2 for k in a["probs"]},
 "geometric mean": lambda a,b: (lambda d: {k:v/sum(d.values()) for k,v in d.items()})(
     {k:(a["probs"][k]*b["probs"][k])**0.5 for k in a["probs"]}),
 "B unless B unsure(<.6)": lambda a,b: b["probs"] if conf(b["probs"])>=0.6 else a["probs"],
 "A unless A unsure(<.6)": lambda a,b: a["probs"] if conf(a["probs"])>=0.6 else b["probs"],
 "rank-normalised conf": lambda a,b: a["probs"] if RA(a["probs"]) >= RB(b["probs"]) else b["probs"],
 "temp-matched conf (A/3)": lambda a,b: a["probs"] if conf(tempered(a["probs"],3.0)) >= conf(b["probs"]) else b["probs"],
 "agree->either, else B": lambda a,b: b["probs"] if top(a["probs"])!=top(b["probs"]) else a["probs"],
 "B, escalate on disagree": lambda a,b: b["probs"],
}
print(f"A = {A_TAG}   B = {B_TAG}   ({len(ids)} shared decisions)\n")
print(f"{'strategy':24s} {'intel':>6s} {'easy':>6s} {'std':>6s} {'hard':>6s} {'ECE':>6s} {'calib':>6s} {'p50':>6s} {'score':>6s}")
print("-"*80)
out={}
for name, fn in STRATS.items():
    per = collections.defaultdict(lambda: [0,0]); pairs=[]; lat=[]
    for i in ids:
        a, b = A[i], B[i]
        p = fn(a, b)
        pred = top(p)
        t = tasks[i]; ok = (pred == str(t.expected))
        d = per[tier(i)]; d[1]+=1; d[0]+=ok
        if tier(i)=="hard": pairs.append((max(p.values()), ok))
        lat.append(a["latency_s"] + b["latency_s"] if "alone" not in name else
                   (a if name.startswith("A") else b)["latency_s"])
    tiers={k:c/n for k,(c,n) in per.items()}
    ece=ece_top_label(pairs)["ece"]
    p50=percentile(sorted(lat),.50); p95=percentile(sorted(lat),.95)
    axes={"intelligence":C.intelligence(tiers),"calibration":C.calibration(ece),
          "speed":C.speed(p50,p95,"gpu"),"cost":C.cost(0.002 if "alone" in name else 0.004)}
    sc=C.jevbench_score(axes)
    print(f"{name:24s} {axes['intelligence']:6.1f} {tiers['easy']:6.3f} {tiers['standard']:6.3f} "
          f"{tiers['hard']:6.3f} {ece:6.3f} {axes['calibration']:6.1f} {p50*1000:6.0f} {sc:6.1f}")
    out[name]={"axes":{k:round(v,1) for k,v in axes.items()},"tiers":{k:round(v,3) for k,v in tiers.items()},
               "ece":round(ece,3),"p50_ms":round(p50*1000),"score":round(sc,1)}
json.dump(out, open("results/jevbench/ensemble.json","w"), indent=1)
# how often does each model win the confidence contest, and who is right?
wa=sum(1 for i in ids if conf(A[i]["probs"])>=conf(B[i]["probs"]))
print(f"\nA wins the confidence contest on {wa}/{len(ids)} decisions ({wa/len(ids)*100:.0f}%)")
for tag,M in (("A",A),("B",B)):
    print(f"  {tag} mean top-confidence: {sum(conf(M[i]['probs']) for i in ids)/len(ids):.3f}")
