"""Analysis for bench/synthetic/PREREGISTRATION.md -- written before the echo-ON runs.

    uv run python bench/synthetic/analyze_echo.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from math import comb

RUNS = {"Gemma 4 E4B": ("syn-gemma-echo-off", "syn-gemma-echo-on"),
        "Qwen3.5-4B": ("syn-qwen-echo-off", "syn-qwen-echo-on")}
FORMS = ("noul", "choice2", "choice4", "choice4r1", "choice4r2")


def load(tag):
    return {r["task_id"]: r for r in map(json.loads, open(f"results/jevbench/{tag}/results.jsonl"))}


def sign_p(fixed: int, broke: int) -> float:
    """Two-sided exact sign test on discordant pairs."""
    n = fixed + broke
    if n == 0:
        return 1.0
    k = min(fixed, broke)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def form_of(tid):
    return tid.rsplit("-", 1)[1]


def scenario_of(tid):
    return tid.rsplit("-", 1)[0]


pooled = defaultdict(lambda: [0, 0])
pooled_h3 = [0, 0]
for model, (off_t, on_t) in RUNS.items():
    off, on = load(off_t), load(on_t)
    assert off.keys() == on.keys()
    print(f"\n=== {model} ===")
    print(f"{'form':10s} {'n':>4s} {'off':>6s} {'on':>6s} {'delta':>6s}  fixed broke  sign p")
    for f in FORMS + ("choice4*",):
        ids = [i for i in off if (form_of(i).startswith("choice4") if f == "choice4*" else form_of(i) == f)]
        a = sum(off[i]["correct"] for i in ids)
        b = sum(on[i]["correct"] for i in ids)
        fx = sum(1 for i in ids if on[i]["correct"] and not off[i]["correct"])
        br = sum(1 for i in ids if off[i]["correct"] and not on[i]["correct"])
        if f in ("noul", "choice2", "choice4*"):
            pooled[f][0] += fx
            pooled[f][1] += br
        print(f"{f:10s} {len(ids):4d} {a/len(ids):6.3f} {b/len(ids):6.3f} {(b-a):+6d}  {fx:5d} {br:5d}  {sign_p(fx, br):.3f}")
    # H3: all three rotations give the same label
    scen = sorted({scenario_of(i) for i in off if form_of(i) == "choice4"})
    def agree(run, s):
        return len({run[f"{s}-{r}"]["predicted"] for r in ("choice4", "choice4r1", "choice4r2")}) == 1
    ag_off = sum(agree(off, s) for s in scen)
    ag_on = sum(agree(on, s) for s in scen)
    up = sum(1 for s in scen if agree(on, s) and not agree(off, s))
    down = sum(1 for s in scen if agree(off, s) and not agree(on, s))
    pooled_h3[0] += up
    pooled_h3[1] += down
    print(f"H3 rotation agreement: off {ag_off}/{len(scen)}  on {ag_on}/{len(scen)}  "
          f"(became consistent {up}, became inconsistent {down}, sign p {sign_p(up, down):.3f})")

print("\n=== pooled over both models (fixed / broke, sign p) ===")
for f, (fx, br) in pooled.items():
    print(f"  {f:9s} fixed {fx:3d}  broke {br:3d}  net {fx-br:+4d}  p {sign_p(fx, br):.3f}")
print(f"  H3 rotations: became consistent {pooled_h3[0]}, inconsistent {pooled_h3[1]}, "
      f"p {sign_p(*pooled_h3):.3f}")
