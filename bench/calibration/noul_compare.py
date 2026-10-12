"""Compare yes/no post-processing variants on served runs, under JevBench v1.6's rule.

BASE is a run of the preset (served probabilities, temperature already applied); CTX the
same yes/no items served with the experimental context bias (dropped: it did not generalise). Fixed offset and floor are applied here
(the server does the same arithmetic, src/ryotide/jevbench_adapter.py _noul_adjust).
A yes/no answer counts only when P(yes) >= 0.8 or <= 0.2; others score as wrong.

    python3 bench/calibration/noul_compare.py BASE/results.jsonl CTX/results.jsonl [--bias -0.75]
"""
import argparse, json, math
from collections import defaultdict


def rows(path):
    out = {}
    for l in open(path):
        r = json.loads(l)
        p = r.get("probs") or {}
        if set(p) == {"yes", "no"}:
            gold_yes = (r["predicted"] == "yes") == bool(r["correct"])
            out[r["task_id"]] = (p["yes"] / (p["yes"] + p["no"]), gold_yes)
    return out


def adjust(py, bias=0.0, floor=None):
    py = min(max(py, 1e-9), 1 - 1e-9)
    py = 1 / (1 + math.exp(-(math.log(py / (1 - py)) + bias)))
    if floor and 1 - floor < py < floor:
        py = floor if py >= 0.5 else 1 - floor
    return py


def score(pairs):
    n = len(pairs); dec = right = yes = 0; brier = 0.0; bins = [[] for _ in range(10)]
    for py, g in pairs:
        top = max(py, 1 - py); ok = (py >= 0.5) == g
        if top >= 0.8 - 1e-12:
            dec += 1; right += ok
        yes += py >= 0.5
        brier += (py - g) ** 2
        bins[min(int(top * 10), 9)].append((top, ok))
    ece = sum(len(b) / n * abs(sum(p for p, _ in b) / len(b) - sum(o for _, o in b) / len(b)) for b in bins if b)
    acc = sum((py >= 0.5) == g for py, g in pairs) / n
    return dict(n=n, acc=acc, said_yes=yes / n, decisive=dec / n, cc=100 * (right / n - 0.5) / 0.5,
                brier=brier / n, ece=ece)


def group(tid):
    if tid.startswith("dec-"):
        return "decide-v1 " + tid.rsplit("-", 1)[1]
    return "hard " + ("hu" if "-hu-" in tid or tid.startswith("hu") else "de" if "-de-" in tid or tid.startswith("de")
                      else "en")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("ctx")
    ap.add_argument("--bias", type=float, default=-0.75, help="fixed offset, fitted before on other items")
    a = ap.parse_args()
    B, C = rows(a.base), rows(a.ctx)
    ids = sorted(set(B) & set(C))
    variants = {
        "preset (now)": lambda i: B[i][0],
        f"fixed bias {a.bias:+.2f}": lambda i: adjust(B[i][0], a.bias),
        "floor 0.8": lambda i: adjust(B[i][0], 0, 0.8),
        f"fixed bias + floor": lambda i: adjust(B[i][0], a.bias, 0.8),
        "context": lambda i: C[i][0],
        "context + floor": lambda i: adjust(C[i][0], 0, 0.8),
    }
    groups = defaultdict(list)
    for i in ids:
        groups[group(i)].append(i); groups["ALL"].append(i)
        groups["ALL decide-v1" if i.startswith("dec-") else "ALL hard"].append(i)
    for g in sorted(groups, key=lambda k: (not k.startswith("ALL"), k)):
        gi = groups[g]
        gold_yes = sum(B[i][1] for i in gi) / len(gi)
        print(f"\n== {g}  (n={len(gi)}, gold yes {gold_yes:.0%})")
        print(f"   {'variant':22s} {'acc':>6} {'saidYes':>8} {'decisive':>9} {'cc':>7} {'brier':>7} {'ece':>6}")
        for name, f in variants.items():
            m = score([(f(i), B[i][1]) for i in gi])
            print(f"   {name:22s} {m['acc']:6.1%} {m['said_yes']:8.0%} {m['decisive']:9.0%} {m['cc']:7.1f} "
                  f"{m['brier']:7.3f} {m['ece']:6.3f}")
