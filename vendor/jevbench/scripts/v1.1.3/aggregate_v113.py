"""JevBench v1.1.3 = the frozen v1.1 task set and v1.1.2 scoring + the GPU round-2 entrants.

    python3 v113/aggregate_v113.py

v1.1.2 rows are copied unchanged from the published v1.1.2 artifact (their items, answers,
prices and scores do not move). Round-2 rows are computed from runs/<key>--easy and
runs/<key>--v1 with the SAME functions v1.1.2 uses (aggregate.block, latency_summary,
jevbench.composite). Then every ranked row is re-ranked under all six weightings.
"""
import json, sys, datetime as dt
from pathlib import Path

JOB = Path(__file__).resolve().parents[1]
V11 = Path.home() / "jobs/jev-class-benchmark-20260919"
sys.path.insert(0, str(V11)); sys.path.insert(0, str(V11 / "repo"))
from aggregate import block  # noqa: E402
from jevbench.tasks import load_jsonl  # noqa: E402
from jevbench.metrics import latency_summary  # noqa: E402
from jevbench.composite import COST_BEST, COST_WORST, HEADLINE, SENSITIVITY, log_score, main_score  # noqa: E402

BASE = json.loads((V11 / "results/v1.1/jevbench-v1.1-results.json").read_text())
assert BASE["revision"] == "v1.1.2", BASE["revision"]
PLAN = {s["key"]: s for s in json.loads((JOB / "bundle/r2_plan.json").read_text())["systems"]}
PRICES = json.loads((JOB / "v113/prices_v113.json").read_text())
HW = json.loads((JOB / "v113/hardware.json").read_text())
TIER_FILES = {"easy": ["repo/datasets/public/easy.jsonl", "private/easy-heldout.jsonl"],
              "standard": ["repo/datasets/public/original.jsonl", "private/heldout.jsonl"],
              "judge": ["private/imported/router.jsonl", "private/imported/judge.jsonl"]}
NET = JOB / "bundle/runs-net"   # ranked runs: from Sandy (Germany) over the internet to the pod
ROUND2 = [k for k in PLAN if (NET / f"{k}--v1").exists() or (NET / f"{k}--easy").exists()]


def rows_of(p):
    f = p / "results.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else None


def entry_for(key, tiers):
    cfg = PLAN[key]
    easy, v1 = rows_of(NET / f"{key}--easy"), rows_of(NET / f"{key}--v1")
    pod_v1 = rows_of(JOB / "runs" / f"{key}--v1") or []
    tb, tr = {}, {}
    for t, ts in tiers.items():
        rows = (easy if t == "easy" else v1) or []
        if rows and not any(r["ok"] for r in rows):
            rows = []
        tr[t] = rows; tb[t] = block(ts, rows)
    all_rows = [r for t in tiers for r in tr[t]]
    coverage = {t: tb[t]["n_attempted"] / len(tiers[t]) for t in tiers}
    accs = [tb[t]["accuracy"] for t in tiers]
    cap = 100 * sum(accs) / 3 if None not in accs else None
    scorable = sum(tb[t]["n_scorable"] for t in tiers if tb[t]["n_attempted"])
    pooled = sum(1 for r in all_rows if r.get("correct")) / scorable if scorable else None
    lat = sorted(r["latency_s"] for r in (v1 or []) if r["ok"] and r.get("latency_s") is not None)
    ls = latency_summary(lat) if lat else {}
    p50, p95 = ls.get("p50_s"), ls.get("p95_s")
    s50, s95 = log_score(p50, 0.1, 10), log_score(p95, 0.1, 10)
    spd = (s50 + s95) / 2 if None not in (s50, s95) else None
    ref = PRICES["systems"][key]
    ins = [r.get("usage", {}).get("input_tokens") for r in all_rows]
    ins = [x for x in ins if isinstance(x, (int, float)) and x > 0]
    tokens_from = None
    if len(ins) < 0.9 * len(all_rows):
        tokens_from = "gemini-3.1-flash-lite"
        src = rows_of(V11 / "private/runs" / tokens_from) + (rows_of(V11 / "private/runs-v1.1" / f"{tokens_from}--easy") or [])
        ins = [r.get("usage", {}).get("input_tokens") for r in src]
        ins = [x for x in ins if isinstance(x, (int, float))]
    mi, mo = sum(ins) / len(ins), ref["out_tokens"]
    cost = 1000 * (mi * ref["in_per_m"] + mo * ref["out_per_m"]) / 1e6
    hw = HW[key]
    e = {
        "key": key, "display": cfg["display"], "class": "jev-rebuild", "open": "yes",
        "author": cfg.get("author"), "repo": cfg.get("repo"), "repo_commit": cfg.get("repo_commit"),
        "licence": cfg.get("licence"), "underlying": cfg.get("underlying"), "note": cfg.get("note"),
        "serving": cfg.get("serving"), "posthoc": bool(cfg.get("posthoc")), "round": "v1.1.3 GPU round",
        "probability_source": sorted({r.get("probs_source", "unknown") for r in all_rows}),
        "has_distribution": any(r.get("probs") for r in all_rows),
        "standard_judge_source": "v1.1.3 run against a rented GPU, from Sandy over the internet",
        "coverage": coverage, "partial": min(coverage.values()) < 0.95,
        "tiers": tb,
        "capability": {"score": cap, "tier_accuracy": dict(zip(tiers, accs)), "pooled_accuracy": pooled,
                       "n_decisions": sum(tb[t]["n_attempted"] for t in tiers)},
        "calibration": {"brier_standard_judge": None, "note": None},
        "speed": {"score": spd, "p50_s": p50, "p95_s": p95, "score_p50": s50, "score_p95": s95, "n": len(lat),
                  "run": "serial 242-decision standard+judge run", "hardware": hw["gpu"],
                  "measured_where": HW[key]["transport"],
                  "gpu_side": _gpu_side(pod_v1, v1)},
        "cost": {"score": log_score(cost, COST_BEST, COST_WORST), "usd_per_1000": cost, "kind": "estimate",
                 "basis": (f"ESTIMATE: hosted-provider price, {ref['ref']} list price ${ref['in_per_m']}/M in, ${ref['out_per_m']}/M out "
                           f"({ref['why']}) x {mi:.0f} input and {mo:.0f} output tokens per decision"
                           + (f" (input tokens counted from the {tokens_from} run, same prompts)" if tokens_from else " (input tokens measured)"))},
        "gpu_rental_for_this_run": hw,
    }
    e["ranked"] = not e["partial"]
    e["calibration"]["brier_standard_judge"] = _brier(tb)
    e["main_score"] = main_score(cap, spd, e["cost"]["score"], SENSITIVITY[HEADLINE])
    e["sensitivity"] = {name: main_score(cap, spd, e["cost"]["score"], w) for name, w in SENSITIVITY.items()}
    return e


def _gpu_side(pod, net):
    """Same 242 decisions run with the client on the GPU machine: latency without the network,
    and how many answers match the ranked network run (determinism check)."""
    if not pod:
        return None
    lat = sorted(r["latency_s"] for r in pod if r["ok"])
    ls = latency_summary(lat)
    a = {r["task_id"]: r.get("predicted") for r in pod}
    same = sum(1 for r in (net or []) if a.get(r["task_id"]) == r.get("predicted"))
    return {"p50_s": ls.get("p50_s"), "p95_s": ls.get("p95_s"), "n": len(lat),
            "same_answer_as_network_run": same, "of": len(net or [])}


def _brier(tb):  # identical to v1.1.2's
    vals = [(tb[t]["brier_mean"], tb[t]["calibration_n"]) for t in ("standard", "judge") if tb[t].get("brier_mean") is not None]
    n = sum(v[1] for v in vals)
    return (sum(v[0] * v[1] for v in vals) / n) if n else None


def main():
    tiers = {t: [x for f in fs for x in load_jsonl(str(V11 / f))] for t, fs in TIER_FILES.items()}
    old = [dict(s) for s in BASE["systems"] if s["key"] != "open-alternative-jev"]
    replaced = next(s for s in BASE["systems"] if s["key"] == "open-alternative-jev")
    for s in old:
        s["round"] = "v1.1"; s.pop("rank_under", None)
        s["speed"] = dict(s["speed"], measured_where=("local, 2 CPU threads of a Ryzen 5 3600" if s["key"] in
                          ("open-jev-deberta-v3-large", "needle-3", "needle-3-tools") else
                          "from a Hetzner server in Germany, network included"))
    new = [entry_for(k, tiers) for k in ROUND2]
    systems = old + new
    ranked = [s for s in systems if s["ranked"] and s["main_score"] is not None]
    for name in SENSITIVITY:
        for i, s in enumerate(sorted(ranked, key=lambda s: -s["sensitivity"][name]), 1):
            s.setdefault("rank_under", {})[name] = i
    systems.sort(key=lambda s: (not s["ranked"], -(s["main_score"] or -1)))
    out = {k: v for k, v in BASE.items() if k not in ("systems",)}
    out.update({
        "benchmark": "JevBench v1.1.3", "protocol": "jevbench::v1.1", "revision": "v1.1.3",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "revision_history": [{"revision": "v1.1.3", "date": "2026-09-19", "note":
            "GPU round: four more open rebuilds run on rented RunPod GPUs (RazorBack16 OpenJev on DiffusionGemma 26B-A4B, SemIf, "
            "open-alternative-jev completed, Bespoke Nimble 9B) on the same 314 frozen v1.1 decisions with the v1.1.2 scoring. "
            "v1.1.2 rows are unchanged; only ranks move. open-alternative-jev's partial v1.1 Space row (6 decisions) is replaced by its complete self-hosted run."}]
            + BASE["revision_history"],
        "revision_note": ("JevBench v1.1.3 (19 Sep 2026) adds entrants that need a GPU. Same items, same answers, same scoring as v1.1.2. "
                          "GPU entrants' Speed is measured on the rented GPU named in each row, with the client on the same machine "
                          "(no internet round trip, model loaded before timing); API entrants' Speed includes the internet from Germany. "
                          "Cost for the GPU entrants is an estimate at hosted-provider list prices (never the GPU rental we paid)."),
        "not_comparable_with": "JevBench v1.0 headline numbers. v1.1.2 rows are identical in v1.1.3; only the cohort and the ranks change.",
        "v11_replaced_rows": {"open-alternative-jev": {"why": "v1.1 had only 6 decisions through the author's free HF Space (GPU quota); "
                              "v1.1.3 ran the same library and model on our rented GPU for all 314.", "v11_n_decisions": replaced["capability"]["n_decisions"]}},
        "systems": systems,
    })
    out["scoring"] = dict(BASE["scoring"], speed=BASE["scoring"]["speed"] + " v1.1.3 GPU entrants: measured on the named rented GPU, client on the same machine.")
    (JOB / "v113/out").mkdir(parents=True, exist_ok=True)
    (JOB / "v113/out/jevbench-v1.1.3-results.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"{'system':32s} {'rk':>3s} {'MAIN':>5s} {'60/20':>5s} {'cap':>5s} {'easy':>5s} {'std':>5s} {'judge':>5s} {'spd':>5s} {'p50':>6s} {'cost':>5s} {'$/1k':>8s}")
    for s in systems:
        ta = s["capability"]["tier_accuracy"]; f = lambda x: f"{x:5.1f}" if isinstance(x, (int, float)) else "    -"
        print(f"{s['key']:32s} {str(s.get('rank_under', {}).get(HEADLINE, '-')):>3s} {f(s['main_score'])} "
              f"{f(s['sensitivity']['60/20/20 accuracy emphasis'] if s.get('sensitivity') else None)} {f(s['capability']['score'])} "
              f"{f(100*ta['easy'] if ta.get('easy') is not None else None)} {f(100*ta['standard'] if ta.get('standard') is not None else None)} "
              f"{f(100*ta['judge'] if ta.get('judge') is not None else None)} {f(s['speed']['score'])} "
              f"{(s['speed']['p50_s'] or 0):6.3f} {f(s['cost']['score'])} {(s['cost']['usd_per_1000'] or 0):8.4f}")


if __name__ == "__main__":
    main()
