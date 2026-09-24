"""Use a second model's AGREEMENT to recalibrate the first model's confidence.

Predictions never change -- the primary model decides. Only its probability is
sharpened when the secondary agrees and flattened when it disagrees. So
intelligence is fixed by construction and the only thing that can move is
calibration (and latency, since both models now run).

Sharpening is done on the odds of the top label, which keeps the distribution
valid and monotone: odds *= k on agreement, odds /= k on disagreement.
"""
import json, sys, collections
sys.path.insert(0, "src")
from jevbench import composite_v13 as C
from jevbench.metrics import ece_top_label, percentile
from jevbench.tasks import load_jsonl

PRIMARY, SECOND = sys.argv[1], sys.argv[2]
tasks = {}
for f in ("original", "easy", "hard"):
    for t in load_jsonl(f"vendor/jevbench/datasets/public/{f}.jsonl"): tasks[t.id] = t
tier = lambda i: "easy" if i.startswith("easy-") else "standard" if i.startswith("original-") else "hard"
load = lambda tag: {r["task_id"]: r for r in (json.loads(l) for l in open(f"results/jevbench/{tag}/results.jsonl"))}
P, S = load(PRIMARY), load(SECOND)
ids = [i for i in P if i in S and P[i].get("probs") and S[i].get("probs")]
top = lambda p: max(p, key=p.get)

agree = {i: top(P[i]["probs"]) == top(S[i]["probs"]) for i in ids}
hard = [i for i in ids if tier(i) == "hard"]
for name, sub in (("all", ids), ("hard tier", hard)):
    ag = [i for i in sub if agree[i]]; dis = [i for i in sub if not agree[i]]
    pa = sum(P[i]["correct"] for i in ag) / max(len(ag), 1)
    pd = sum(P[i]["correct"] for i in dis) / max(len(dis), 1)
    print(f"{name:10s} agree {len(ag):3d} -> primary correct {pa:.3f} | disagree {len(dis):3d} -> {pd:.3f}"
          f"   separation {pa-pd:+.3f}")

def reweight(p, k):
    """Scale the odds of the top label by k, keep the rest proportional."""
    t = top(p); q = p[t]
    if q >= 0.999999 or len(p) < 2: return dict(p)
    o = q / (1 - q) * k
    nq = o / (1 + o); scale = (1 - nq) / (1 - q)
    return {kk: (nq if kk == t else v * scale) for kk, v in p.items()}

per = collections.defaultdict(lambda: [0, 0])
for i in ids:
    d = per[tier(i)]; d[1] += 1; d[0] += bool(P[i]["correct"])
tiers = {k: c / n for k, (c, n) in per.items()}
intel = C.intelligence(tiers)
lat_solo = sorted(P[i]["latency_s"] for i in ids)
lat_both = sorted(P[i]["latency_s"] + S[i]["latency_s"] for i in ids)

print(f"\n{'k (odds factor)':18s} {'hard ECE':>9s} {'calib':>7s} {'speed':>7s} {'score':>7s}")
print("-" * 54)
rows = {}
for label, k, lat in [("1.0 (primary alone)", 1.0, lat_solo)] + [(f"{k}", k, lat_both) for k in (1.5, 2, 3, 5, 8)]:
    pairs = []
    for i in hard:
        p = P[i]["probs"] if k == 1.0 else reweight(P[i]["probs"], k if agree[i] else 1 / k)
        pairs.append((max(p.values()), bool(P[i]["correct"])))
    ece = ece_top_label(pairs)["ece"]
    axes = {"intelligence": intel, "calibration": C.calibration(ece),
            "speed": C.speed(percentile(lat, .5), percentile(lat, .95), "gpu"),
            "cost": C.cost(0.002 if k == 1.0 else 0.004)}
    sc = C.jevbench_score(axes)
    print(f"{label:18s} {ece:9.3f} {axes['calibration']:7.1f} {axes['speed']:7.1f} {sc:7.1f}")
    rows[label] = {"ece": round(ece, 3), "calibration": round(axes["calibration"], 1), "score": round(sc, 1)}
print(f"\nintelligence is fixed at {intel:.1f} for every row -- predictions never change.")
json.dump(rows, open("results/jevbench/agree_boost.json", "w"), indent=1)
