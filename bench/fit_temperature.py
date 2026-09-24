"""Fit the softmax temperature over the option markers -- on non-benchmark data only.

    uv run python bench/fit_temperature.py --fit <synthetic-run-tag> [--report <tag> ...]

--fit     a run on data/synthetic/*.jsonl made with --temperature 1. Scenarios are
          split by index: even ones fit T (minimum negative log-likelihood of the
          gold option), odd ones check it. JevBench items are never used to fit.
--report  any other runs made with --temperature 1 (e.g. the public JevBench items):
          their calibration is REPORTED at the fitted T, never used to choose it.

Temperature only rescales confidence; it cannot change which option wins, so
accuracy is identical at every T.

Decision rule (fixed before the CUDA runs): the fitted T is ADOPTED only if it
lowers the expected calibration error on the held-out (odd) synthetic half;
otherwise T = 1 is kept. No JevBench item takes part in the decision.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "jevbench"))
from jevbench.metrics import ece_top_label  # noqa: E402


def load(tag):
    rows = [json.loads(l) for l in open(os.path.join("results", "jevbench", tag, "results.jsonl"))]
    temps = {(r.get("runtime") or {}).get("temperature", 1.0) for r in rows}
    if temps - {1.0, None}:
        sys.exit(f"{tag}: made with temperature {temps}; fit and report need --temperature 1 runs")
    return [r for r in rows if r.get("probs")]


def rescale(probs: dict, t: float) -> dict:
    z = {k: math.log(max(v, 1e-300)) / t for k, v in probs.items()}
    top = max(z.values())
    e = {k: math.exp(v - top) for k, v in z.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}


def nll(rows, t, labels):
    return sum(-math.log(max(rescale(r["probs"], t)[labels[r["task_id"]]], 1e-300)) for r in rows) / len(rows)


def ece(rows, t):
    pairs = []
    for r in rows:
        p = rescale(r["probs"], t)
        top = max(p, key=p.get)
        pairs.append((p[top], top == r["predicted"] and bool(r["correct"])))
    return ece_top_label(pairs)["ece"]


def fit(rows, labels):
    lo, hi = math.log(0.2), math.log(20.0)          # golden-section search on log T
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = nll(rows, math.exp(a), labels), nll(rows, math.exp(b), labels)
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo); fa = nll(rows, math.exp(a), labels)
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo); fb = nll(rows, math.exp(b), labels)
    return math.exp((lo + hi) / 2)


def gold_labels(task_files):
    labels = {}
    for f in task_files:
        for line in open(f):
            d = json.loads(line)
            labels[d["id"]] = str(d["expected"])
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--report", nargs="*", default=[])
    ap.add_argument("--tasks", nargs="*", default=["data/synthetic/echo-v1.jsonl",
                    "vendor/jevbench/datasets/public/original.jsonl",
                    "vendor/jevbench/datasets/public/easy.jsonl",
                    "vendor/jevbench/datasets/public/hard.jsonl"])
    a = ap.parse_args()
    labels = gold_labels(a.tasks)
    rows = load(a.fit)
    scen = lambda r: int(r["task_id"].split("-")[2])
    fit_rows = [r for r in rows if scen(r) % 2 == 0]
    chk_rows = [r for r in rows if scen(r) % 2 == 1]
    t = fit(fit_rows, labels)
    print(f"fitted on {a.fit}: {len(fit_rows)} items (even scenarios);  T = {t:.3f}")
    print(f"{'set':34s} {'n':>4s}  {'NLL T=1':>8s} {'NLL T':>8s}  {'ECE T=1':>8s} {'ECE T':>7s}")
    for name, rs in ((f"{a.fit} (fit half)", fit_rows), (f"{a.fit} (check half)", chk_rows)):
        print(f"{name:34s} {len(rs):4d}  {nll(rs, 1, labels):8.3f} {nll(rs, t, labels):8.3f}  "
              f"{ece(rs, 1):8.3f} {ece(rs, t):7.3f}")
    for tag in a.report:
        rs = load(tag)
        hard = [r for r in rs if r["task_id"].startswith("hard-")]
        for name, sub in ((tag, rs), (f"{tag} [hard]", hard)):
            if sub:
                print(f"{name[:34]:34s} {len(sub):4d}  {nll(sub, 1, labels):8.3f} {nll(sub, t, labels):8.3f}  "
                      f"{ece(sub, 1):8.3f} {ece(sub, t):7.3f}   (report only)")
    before, after = ece(chk_rows, 1), ece(chk_rows, t)
    adopt = after < before
    print(f"\ndecision: check-half ECE {before:.3f} -> {after:.3f}: "
          f"{'ADOPT the fitted T' if adopt else 'KEEP T = 1 (fit does not generalise)'}")
    print(f"T = {t if adopt else 1.0:.3f}")


if __name__ == "__main__":
    main()
