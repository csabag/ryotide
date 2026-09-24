"""GLUE sweep. Phase A: prompt/calibration config. Phase B: model variant."""
import json, sys, time, argparse, os
sys.path.insert(0, "src")
from ryotide.model import load_model, describe
from ryotide.classify import Classifier
from ryotide.glue import evaluate, TASKS

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["mlx-community/Qwen2.5-0.5B-Instruct-4bit"])
ap.add_argument("--tasks", nargs="+", default=["sst2", "rte"])
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--configs", nargs="+", default=["chat+cal","chat","raw+cal","raw"])
ap.add_argument("--out", default="results/glue.json")
a = ap.parse_args()

rows = []
for mname in a.models:
    t0=time.time(); model, tok = load_model(mname); clf = Classifier(model, tok)
    info = describe(model, mname)
    print(f"\n=== {mname}  ({time.time()-t0:.0f}s load, {info.n_layers}L, kv {info.kv_bytes_per_token}B/tok)", flush=True)
    for task in a.tasks:
        for cfg in a.configs:
            chat = cfg.startswith("chat"); cal = cfg.endswith("cal")
            try:
                r = evaluate(clf, TASKS[task], n=a.n, chat=chat, calibrate=cal)
            except Exception as e:
                print(f"  {task:6s} {cfg:9s} FAILED: {type(e).__name__}: {e}", flush=True); continue
            r["model"] = mname; r["config"] = cfg
            r["kv_bytes_per_token"] = info.kv_bytes_per_token
            rows.append(r)
            extra = f" mcc={r['matthews']}" if "matthews" in r else (f" f1={r['f1']}" if "f1" in r else "")
            print(f"  {task:6s} {cfg:9s} acc={r['accuracy']:.3f} (maj {r['majority_baseline']:.3f}){extra}"
                  f"  {r['ms_per_example']:.0f}ms/ex  pred={r['pred_distribution']}", flush=True)
    del model, clf
    os.makedirs("results", exist_ok=True)
    json.dump(rows, open(a.out,"w"), indent=1)
json.dump(rows, open(a.out,"w"), indent=1)
print(f"\nwrote {a.out} ({len(rows)} rows)")
