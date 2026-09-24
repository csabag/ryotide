"""Compare our runs against published JevBench systems on the SAME tiers.

The published board scores four tiers; we only have three (judge is held out).
Rather than compare our 3-tier number to their 4-tier number, recompute theirs
on our three tiers using the harness's own renormalising intelligence(), so both
sides are measured over identical ground.
"""
import json, sys
sys.path.insert(0, "src")
from jevbench import composite_v13 as C

PUB = json.load(open("vendor/jevbench/results/v1.2/jevbench-v1.2-results.json"))
OURS = {r["tag"]: r for r in json.load(open("results/jevbench/scores.json"))}
OUR_TIERS = ("easy", "standard", "hard")

rows = []
for s in PUB["systems"]:
    t = s.get("tiers") or {}
    if not s.get("jevbench_score") or not all(k in t for k in OUR_TIERS):
        continue
    sub = {k: t[k] for k in OUR_TIERS}
    rows.append({"name": s["display"], "ours": False, "published": s["jevbench_score"],
                 "intel_3tier": C.intelligence(sub), "tiers": sub,
                 "ranked": s.get("ranked", True), "open": s.get("open")})
for tag, r in OURS.items():
    t = r["tiers"]
    rows.append({"name": tag, "ours": True, "published": None,
                 "intel_3tier": C.intelligence(t), "tiers": t, "ranked": True, "open": True})

rows.sort(key=lambda r: -r["intel_3tier"])
print(f"{'system':40s} {'intel':>6s} {'easy':>6s} {'std':>6s} {'hard':>6s}  {'pub 4-tier':>10s}")
print("-" * 82)
for r in rows:
    if not r["ours"] and r["intel_3tier"] < 20:
        continue
    mark = " <<<" if r["ours"] else ("  (unranked)" if not r["ranked"] else "")
    pub = f"{r['published']:.1f}" if r["published"] else "--"
    t = r["tiers"]
    print(f"{r['name'][:40]:40s} {r['intel_3tier']:6.1f} {t['easy']:6.3f} "
          f"{t['standard']:6.3f} {t['hard']:6.3f}  {pub:>10s}{mark}")
json.dump(rows, open("results/jevbench/board_compare.json", "w"), indent=1)
