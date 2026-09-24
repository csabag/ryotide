"""Offline sweep over trajectory combination rules. usage: traj_analyze.py TAG
Note: UNGATED accumulation telescopes. From p0 it equals the final trajectory read;
from uniform it equals p_T / p_0 -- i.e. contextual calibration by the no-evidence prior.
Gating is the only thing that makes accumulation a new rule."""
import sys, json, math
R = [json.loads(l) for l in open(f"results/probes/traj_{sys.argv[1]}_items.jsonl")]
# Validity filter, fixed BEFORE looking at any rule result: a checkpoint read whose
# marker mass is < 0.9 did not land on an answer position (state cut mid-table/code),
# and would masquerade as a huge belief swing. Drop it; p0 is kept regardless.
MASS_MIN = 0.9
_dropped = _total = 0
for r in R:
    keep = [0] + [i for i in range(1, len(r["traj"])) if r["traj_mass"][i] >= MASS_MIN]
    _dropped += len(r["traj"]) - len(keep); _total += len(r["traj"])
    r["traj"] = [r["traj"][i] for i in keep]; r["traj_mass"] = [r["traj_mass"][i] for i in keep]
print(f"validity filter: dropped {_dropped} of {_total} checkpoints with marker mass < {MASS_MIN}")
n = len(R); L = lambda p: [math.log(max(x, 1e-12)) for x in p]
am = lambda v: max(range(len(v)), key=lambda j: v[j])
acc = lambda f: sum(am(f(r)) == r["gold"] for r in R) / n
def kl(p, q): return sum(a * (math.log(max(a,1e-12)) - math.log(max(b,1e-12))) for a, b in zip(p, q))

def gated(r, tau, start):
    T = r["traj"]; s = [0.0]*r["k"] if start == "uniform" else L(T[0])
    for i in range(1, len(T)):
        if kl(T[i], T[i-1]) > tau:
            s = [a + b - c for a, b, c in zip(s, L(T[i]), L(T[i-1]))]
    return s

print(f"n={n}   base (S/Q/Q) {acc(lambda r: r['base']):.3f}   p0 (no state) {acc(lambda r: r['traj'][0]):.3f}"
      f"   traj-final (Q/S) {acc(lambda r: r['traj'][-1]):.3f}   p_T/p_0 {acc(lambda r: [a-b for a,b in zip(L(r['traj'][-1]),L(r['traj'][0]))]):.3f}")
print(f"marker mass: base min {min(r['base_mass'] for r in R):.3f}   trajectory min {min(min(r['traj_mass']) for r in R):.3f}")
print("\ngated accumulation (update only when KL(p_t||p_t-1) > tau):")
for start in ("p0", "uniform"):
    print(f"  from {start:7s} " + "  ".join(f"tau={t}: {acc(lambda r: gated(r, t, start)):.3f}" for t in (0, 0.01, 0.05, 0.1, 0.3, 1.0)))
print("\nblend log(base) + w * log(traj-final):")
print("  " + "  ".join(f"w={w}: {acc(lambda r: [a + w*b for a, b in zip(L(r['base']), L(r['traj'][-1]))]):.3f}" for w in (0, 0.25, 0.5, 1.0, 2.0)))

# -- nudge the S/Q/Q decision with the trajectory (w = 0 is the baseline in every row)
def evid(r, tau):              # sum of the per-chunk updates that cleared the gate
    T = r["traj"]; s = [0.0]*r["k"]
    for i in range(1, len(T)):
        if kl(T[i], T[i-1]) > tau:
            s = [a + b - c for a, b, c in zip(s, L(T[i]), L(T[i-1]))]
    return s
mean_traj = lambda r: [sum(col)/len(col) for col in zip(*[L(p) for p in r["traj"]])]
net = lambda r: [a - b for a, b in zip(L(r["traj"][-1]), L(r["traj"][0]))]
W = (0, 0.1, 0.25, 0.5, 1.0)
nudge = lambda extra, w: (lambda r: [a + w*b for a, b in zip(L(r["base"]), extra(r))])
print("\nnudging the S/Q/Q decision:  log p_base + w * <trajectory term>")
print(f"  {'term':28s} " + "  ".join(f"w={w:<4}" for w in W))
rows = [("trajectory mean", mean_traj), ("net evidence (p_T/p_0)", net)] + \
       [(f"gated evidence tau={t}", (lambda t: (lambda r: evid(r, t)))(t)) for t in (0.01, 0.05, 0.1, 0.3)]
for name, f in rows:
    print(f"  {name:28s} " + "  ".join(f"{acc(nudge(f, w)):.3f} " for w in W))
print(f"  ({len(rows)*(len(W)-1)} non-trivial settings on {n} items: pick here, CONFIRM on the held-out 71)")

# -- side-LSTM with a hand-set FORGET gate: ignore small updates, add moderate ones,
#    and on a very large one REPLACE the cell with the current belief (a supersession
#    looks like a big mid-document swing). No learned parameters.
def lstm(r, tau, reset):
    T = r["traj"]; cell = L(T[0])
    for i in range(1, len(T)):
        d = kl(T[i], T[i-1])
        if d > reset:  cell = L(T[i])                                          # forget + write
        elif d > tau:  cell = [a + b - c for a, b, c in zip(cell, L(T[i]), L(T[i-1]))]  # input gate open
    return cell
print("\nside-LSTM (forget gate = reset when KL > r), as a nudge on S/Q/Q:  log p_base + w * cell")
for tau, rs in ((0.05, 0.5), (0.05, 1.0), (0.1, 1.0), (0.1, 2.0)):
    print(f"  tau={tau:<4} reset={rs:<4} " + "  ".join(f"w={w}: {acc(nudge((lambda t, q: (lambda r: lstm(r, t, q)))(tau, rs), w)):.3f}" for w in W))
resets = [sum(kl(r['traj'][i], r['traj'][i-1]) > 1.0 for i in range(1, len(r['traj']))) for r in R]
print(f"  items with >=1 reset at r=1.0: {sum(x > 0 for x in resets)}/{n}")

# -- does trajectory instability flag baseline errors that confidence cannot?
def auroc(score, err):          # P(score of an error > score of a correct)
    e = [s for s, x in zip(score, err) if x]; c = [s for s, x in zip(score, err) if not x]
    return sum((a > b) + 0.5*(a == b) for a in e for b in c) / max(1, len(e)*len(c))
err = [am(r["base"]) != r["gold"] for r in R]
feats = {
  "1 - base confidence": [1 - max(r["base"]) for r in R],
  "argmax flips":        [sum(am(r["traj"][i]) != am(r["traj"][i-1]) for i in range(1, len(r["traj"]))) for r in R],
  "late flip (pos/T)":   [max([i for i in range(1, len(r["traj"])) if am(r["traj"][i]) != am(r["traj"][i-1])] or [0]) / max(1, len(r["traj"])-1) for r in R],
  "base vs traj disagree": [float(am(r["base"]) != am(r["traj"][-1])) for r in R],
  "max drop of base answer along traj": [max(0, max(p[am(r['base'])] for p in r["traj"]) - r["traj"][-1][am(r["base"])]) for r in R],
}
print(f"\npredicting baseline errors ({sum(err)} of {n}); AUROC 0.5 = useless:")
for k_, v in feats.items(): print(f"  {k_:36s} {auroc(v, err):.3f}")
hi = [i for i, r in enumerate(R) if max(r["base"]) >= 0.98 and err[i]]
print(f"\nconfident errors (base conf >= 0.98): {len(hi)} -> flips {[feats['argmax flips'][i] for i in hi]}, "
      f"base/traj disagree {[int(feats['base vs traj disagree'][i]) for i in hi]}")
