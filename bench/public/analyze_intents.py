"""Report for the public intent runs.

    uv run python bench/public/analyze_intents.py pub-gemma-banking77 pub-gemma-clinc150 ...
"""
import json
import sys

print(f"{'run':24s} {'n':>5s} {'k':>4s} {'acc':>6s} {'chance':>7s} {'invalid':>7s} {'mass p5':>8s} "
      f"{'p50 ms':>7s}  notes")
for tag in sys.argv[1:]:
    R = [json.loads(l) for l in open(f"results/jevbench/{tag}/results.jsonl")]
    gold = {d["id"]: d["expected"] for d in map(json.loads, open(f"data/public/{tag.split('-', 2)[2]}.jsonl"))}
    for r in R:
        r["expected"] = gold[r["task_id"]]
    ok = [r for r in R if r["ok"]]
    k = len(ok[0]["probs"]) if ok and ok[0].get("probs") else 0
    mass = sorted(r["runtime"]["marker_mass"] for r in ok)
    lat = sorted(r["latency_s"] for r in ok)
    note = ""
    if "clinc" in tag:
        ins = [r for r in R if r.get("expected") != "oos"]
        oos = [r for r in R if r.get("expected") == "oos"]
        if oos:
            note = (f"in-scope {sum(r['correct'] for r in ins)/len(ins):.3f}, "
                    f"oos recall {sum(r['correct'] for r in oos)/len(oos):.3f}")
    print(f"{tag:24s} {len(R):5d} {k:4d} {sum(r['correct'] for r in R)/len(R):6.3f} "
          f"{(1/k if k else 0):7.3f} {len(R)-len(ok):7d} {mass[len(mass)//20]:8.3f} "
          f"{lat[len(lat)//2]*1000:7.0f}  {note}")
