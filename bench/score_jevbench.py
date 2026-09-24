"""Turn our JevBench runs into JevBench Scores using the harness's own v1.3 scorer.

IMPORTANT -- this is a PUBLIC-SUBSET score, not a board-comparable one:
  * 231 of 534 decisions are public. The judge tier (146) is entirely held out,
    so intelligence() renormalises over easy/standard/hard only.
  * Calibration uses ECE alone; the gold distributions the full score also
    rewards ship only with the held-out set.
  * Cost has no provider tariff for local weights. We supply an energy-based
    estimate and label it as ours -- the board estimates by size class instead.
"""
import json, sys
sys.path.insert(0, "src")
from jevbench import composite_v13 as C
from jevbench.metrics import ece_top_label, percentile

TIER_FILES = {"easy": "easy", "standard": "original", "hard": "hard"}
WATTS, PRICE_KWH = 30.0, 0.30   # whole-machine draw under GPU load; EU domestic tariff

def tier_of(task_id):
    for tier, prefix in TIER_FILES.items():
        if task_id.startswith(prefix + "-"):
            return tier
    return None

rows = []
for tag in sys.argv[1:] or ["1.5b-o1", "1.5b-o2", "0.5b-o2"]:
    try:
        rs = [json.loads(l) for l in open(f"results/jevbench/{tag}/results.jsonl")]
    except FileNotFoundError:
        print(f"  ({tag}: no results yet, skipping)"); continue
    if len(rs) < 200:
        print(f"  ({tag}: only {len(rs)} decisions, still running - skipping)"); continue
    tiers, pairs_hard, lat = {}, [], []
    per = {}
    for r in rs:
        t = tier_of(r["task_id"])
        per.setdefault(t, [0, 0])
        per[t][1] += 1; per[t][0] += bool(r["correct"])
        lat.append(r["latency_s"])
        if t == "hard" and r.get("probs"):
            p = r["probs"]; top = max(p, key=p.get)
            pairs_hard.append((p[top], bool(r["correct"])))
    tiers = {t: c / n for t, (c, n) in per.items() if t}

    ece = ece_top_label(pairs_hard)["ece"]
    p50, p95 = percentile(sorted(lat), .50), percentile(sorted(lat), .95)
    mean_s = sum(lat) / len(lat)
    usd_1k = mean_s * 1000 / 3600 * WATTS / 1000 * PRICE_KWH

    axes = {
        "intelligence": C.intelligence(tiers),
        "calibration": C.calibration(ece, mean_tvd=None),
        "speed": C.speed(p50, p95, "gpu"),
        "cost": C.cost(usd_1k),
    }
    score = C.jevbench_score(axes)
    rows.append((tag, tiers, axes, score, p50, p95, usd_1k, ece,
                 {k: C.preset_score(axes, w) for k, w in C.PRESETS.items()}))

print(f"chance baselines: " + ", ".join(f"{k} {v:.3f}" for k, v in C.TIER_CHANCES.items()))
print(f"speed adjustment for self-hosted GPU: latency x{C.LOAD_FACTOR} + {C.OWN_SERVER_ADD_S}s\n")
for tag, tiers, axes, score, p50, p95, usd, ece, presets in rows:
    print(f"=== {tag} ===")
    print("  tier accuracy  " + "  ".join(f"{k} {v:.3f}" for k, v in sorted(tiers.items())) + "   (judge: not public)")
    print(f"  raw p50 {p50*1000:.0f}ms  p95 {p95*1000:.0f}ms   hard-tier ECE {ece:.4f}   est ${usd:.5f}/1k decisions")
    print("  axes  " + "  ".join(f"{k} {v:.1f}" for k, v in axes.items()))
    print(f"  near-chance multiplier {C.near_chance_multiplier(axes['intelligence']):.3f}")
    print(f"  >> JevBench Score (public subset) {score:.1f}")
    for k, v in presets.items():
        if k != C.MAIN:
            print(f"       {k:38s} {v:5.1f}")
    print()
json.dump([{ "tag": r[0], "tiers": r[1], "axes": r[2], "score": r[3],
             "p50_s": r[4], "p95_s": r[5], "usd_per_1k": r[6], "hard_ece": r[7],
             "presets": r[8]} for r in rows],
          open("results/jevbench/scores.json", "w"), indent=1)
print("wrote results/jevbench/scores.json")
