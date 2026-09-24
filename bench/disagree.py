"""When a cheaper model CONFIDENTLY disagrees with the big one, who is right?

That is the only pattern that makes a cheap second opinion worth running: the
small model must be both right and confident precisely where the big one errs.
"""
import json, sys
sys.path.insert(0, "src")
from jevbench.tasks import load_jsonl
tasks = {}
for f in ("original", "easy", "hard"):
    for t in load_jsonl(f"vendor/jevbench/datasets/public/{f}.jsonl"): tasks[t.id] = t
tier = lambda i: "easy" if i.startswith("easy-") else "standard" if i.startswith("original-") else "hard"
load = lambda t: {r["task_id"]: r for r in (json.loads(l) for l in open(f"results/jevbench/{t}/results.jsonl"))}
top = lambda p: max(p, key=p.get); conf = lambda p: max(p.values())

BIG = sys.argv[1] if len(sys.argv) > 1 else "gemma4-e4b-8bit-o1"
SMALL = sys.argv[2:] or ["3.5-4b-8bit-o1","gemma4-e2b-8bit-o1","3.5-2b-o1","1.5b-o4","0.5b-8bit-base"]
B = load(BIG)
print(f"big model: {BIG}\n")
print(f"{'small model':22s} {'thr':>5s} {'confident disagreements':>23s} {'small right':>12s} {'big right':>10s} {'verdict':>10s}")
print("-" * 92)
for tag in SMALL:
    if tag == BIG: continue
    try: S = load(tag)
    except FileNotFoundError: continue
    ids = [i for i in S if i in B and S[i].get("probs") and B[i].get("probs")]
    for thr in (0.0, 0.7, 0.9, 0.95, 0.99):
        d = [i for i in ids if top(S[i]["probs"]) != top(B[i]["probs"]) and conf(S[i]["probs"]) >= thr]
        if not d: continue
        sw = sum(1 for i in d if S[i]["correct"]); bw = sum(1 for i in d if B[i]["correct"])
        v = "SMALL wins" if sw > bw else ("tie" if sw == bw else "big wins")
        print(f"{tag:22s} {thr:5.2f} {len(d):23d} {sw:9d} ({sw/len(d)*100:3.0f}%) {bw:6d} ({bw/len(d)*100:3.0f}%) {v:>10s}")
    # where small is right and big is wrong -- is small confident there?
    win = [i for i in ids if S[i]["correct"] and not B[i]["correct"]]
    lose = [i for i in ids if not S[i]["correct"] and B[i]["correct"]]
    if win:
        mw = sum(conf(S[i]["probs"]) for i in win)/len(win)
        ml = sum(conf(S[i]["probs"]) for i in lose)/max(len(lose),1)
        hard_win = sum(1 for i in win if tier(i)=="hard")
        print(f"{'':22s}   small-right/big-wrong: {len(win):3d} (hard {hard_win})  mean conf {mw:.3f}"
              f"   vs small-wrong/big-right: {len(lose):3d}  mean conf {ml:.3f}   gap {mw-ml:+.3f}")
    print()
