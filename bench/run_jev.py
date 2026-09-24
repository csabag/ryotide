"""Run the real Jev through JevBench's own TypeSafe adapter, via OpenRouter.

OpenRouter serves Jev on a Decisions API (/api/v1/systemone), which is the same
shape the harness's TypeSafeAdapter already speaks -- so we use their adapter
unmodified and only point it at OpenRouter's host. Nothing about the request or
the scoring is ours.
"""
import argparse, json, os, shutil, sys, time
sys.path.insert(0, "src")
from jevbench.adapters import TypeSafeAdapter
from jevbench.budget import Ledger
from jevbench.runner import Runner
from jevbench.summarize import summarize
from jevbench.tasks import load_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", default=",".join(
    f"vendor/jevbench/datasets/public/{f}.jsonl" for f in ("original","easy","hard")))
ap.add_argument("--model", default="typesafe/jev-1.13")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--tag", default="jev-real")
ap.add_argument("--cap-usd", type=float, default=0.50)
a = ap.parse_args()

for line in open(".env"):                      # key stays in the process only
    if line.strip() and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
if not os.environ.get("OPENROUTER_API_KEY"):
    sys.exit("OPENROUTER_API_KEY not found in .env")

tasks = [t for f in a.tasks.split(",") for t in load_jsonl(f.strip())]
if a.limit: tasks = tasks[:a.limit]
out = f"results/jevbench/{a.tag}"; shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
raw = f"/tmp/jev-raw/{a.tag}"; shutil.rmtree(raw, ignore_errors=True)

ad = TypeSafeAdapter(endpoint="https://openrouter.ai/api", model=a.model,
                     key_env="OPENROUTER_API_KEY",
                     price_input_per_m=0.042, price_output_per_m=0.0)
print(f"[jev] endpoint={ad.endpoint}/v1/systemone model={ad.model} tasks={len(tasks)}", flush=True)
runner = Runner(ad, Ledger(f"{out}/ledger.jsonl", cap_usd=a.cap_usd), raw_dir=raw,
                default_reserve_usd=0.0001)
t0 = time.time()
records = runner.run_all(tasks, progress_every=25, results_path=f"{out}/results.jsonl")
json.dump(summarize(tasks, records), open(f"{out}/summary.json","w"), indent=1, default=str)
ok = [r for r in records if r["ok"]]
acc = sum(1 for r in records if r["correct"]) / max(len(records),1)
cost = sum((r.get("cost_usd") or 0) for r in records)
lat = sorted(r["latency_s"] for r in ok)
print(f"\n=== {a.tag} ===  n={len(records)} ok={len(ok)} valid={sum(1 for r in records if r['valid'])}")
print(f"  accuracy {acc:.3f}   cost ${cost:.4f}   wall {time.time()-t0:.0f}s")
if lat: print(f"  latency p50 {lat[len(lat)//2]*1000:.0f}ms  p95 {lat[int(len(lat)*.95)-1]*1000:.0f}ms")
