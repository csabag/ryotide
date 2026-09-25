"""Letters vs digit labels (1-9). Synthetic decides; JevBench public is secondary."""
import json, sys
from math import comb
sys.path.insert(0, "vendor/jevbench")
from jevbench.metrics import ece_top_label
load = lambda t: {r["task_id"]: r for r in map(json.loads, open(f"results/jevbench/{t}/results.jsonl"))}
def sign_p(a, b):
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)
qtype = {}
for f in ("data/synthetic/echo-v1.jsonl", "vendor/jevbench/datasets/public/original.jsonl",
          "vendor/jevbench/datasets/public/easy.jsonl", "vendor/jevbench/datasets/public/hard.jsonl"):
    for l in open(f):
        d = json.loads(l); qtype[d["id"]] = d["question"]["type"]
pairs = [("SYNTHETIC", "Gemma", "syn-gemma-echo-on", "syn-gemma-digits"),
         ("SYNTHETIC", "Qwen", "syn-qwen-echo-on", "syn-qwen-digits"),
         ("JevBench public (secondary)", "Gemma", "gemma4-rep2-pin2", "pub-gemma-digits"),
         ("JevBench public (secondary)", "Qwen", "3.5-4b-rep2", "pub-qwen-digits")]
for setname, model, lt, dt in pairs:
    try: L, D = load(lt), load(dt)
    except FileNotFoundError: print(f"{setname} {model}: pending"); continue
    ids = sorted(L.keys() & D.keys())
    print(f"\n{setname} · {model}  (n={len(ids)})")
    for name, sub in (("all", ids), ("yes/no", [i for i in ids if qtype[i] == "noul"]),
                      ("choice", [i for i in ids if qtype[i] == "choice"])):
        if not sub: continue
        fx = sum(1 for i in sub if D[i]["correct"] and not L[i]["correct"])
        br = sum(1 for i in sub if L[i]["correct"] and not D[i]["correct"])
        ece = lambda R: ece_top_label([(max(R[i]["probs"].values()), bool(R[i]["correct"])) for i in sub])["ece"]
        md = min(D[i]["runtime"]["marker_mass"] for i in sub)
        print(f"  {name:7s} letters {sum(L[i]['correct'] for i in sub):3d}  digits {sum(D[i]['correct'] for i in sub):3d}"
              f"  | fixed {fx:2d} broke {br:2d}  sign p {sign_p(fx, br):.3f}"
              f"  | ECE {ece(L):.3f} -> {ece(D):.3f}  | digit mass min {md:.3f}")
