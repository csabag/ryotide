"""Did the echo transfer to Jev? Written before the runs finished.

An effect counts only if it beats Jev's own run-to-run noise:
  noise   = disagreements base-1 vs base-2, and echo-1 vs echo-2
  control = JSON-state items, sent unchanged even in the echo runs
  effect  = items CONSISTENTLY flipped: wrong in both base runs and right in both
            echo runs ("fixed"), or the reverse ("broke"); exact sign test on those.
"""
import json
from math import comb

load = lambda t: {r["task_id"]: r for r in map(json.loads, open(f"results/jevbench/{t}/results.jsonl"))}
B1, B2, E1, E2 = (load(t) for t in ("jev-base-1", "jev-base-2", "jev-echo-1", "jev-echo-2"))
state_kind = {}
tier = {}
for f in ("original", "easy", "hard"):
    for line in open(f"vendor/jevbench/datasets/public/{f}.jsonl"):
        d = json.loads(line)
        state_kind[d["id"]] = "text" if isinstance(d["state"], str) else "json"
        tier[d["id"]] = f
ids = sorted(B1.keys() & B2.keys() & E1.keys() & E2.keys())


def sign_p(a, b):
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def disagree(X, Y, sub):
    return sum(X[i]["predicted"] != Y[i]["predicted"] for i in sub)


print(f"items in all four runs: {len(ids)}  (failed/invalid excluded: "
      f"{sum(not r['ok'] for R in (B1, B2, E1, E2) for r in R.values())})")
print(f"models: {sorted({r.get('model') for R in (B1, B2, E1, E2) for r in R.values()})}\n")
for name, sub in (("all", ids), ("text states (echo applied)", [i for i in ids if state_kind[i] == "text"]),
                  ("JSON states (never changed)", [i for i in ids if state_kind[i] == "json"]),
                  ("hard tier", [i for i in ids if tier[i] == "hard"])):
    acc = lambda R: sum(R[i]["correct"] for i in sub)
    fx = sum(1 for i in sub if not B1[i]["correct"] and not B2[i]["correct"] and E1[i]["correct"] and E2[i]["correct"])
    br = sum(1 for i in sub if B1[i]["correct"] and B2[i]["correct"] and not E1[i]["correct"] and not E2[i]["correct"])
    print(f"{name:28s} n={len(sub):3d} | correct base {acc(B1)}, {acc(B2)} | echo {acc(E1)}, {acc(E2)} | "
          f"noise: base-vs-base {disagree(B1, B2, sub)}, echo-vs-echo {disagree(E1, E2, sub)} | "
          f"consistent: fixed {fx} broke {br} (sign p {sign_p(fx, br):.3f})")
