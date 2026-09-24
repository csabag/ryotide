"""Run JevBench against the MLX classifier using the harness's own Runner,
Ledger and scoring -- we supply only the adapter."""
import argparse, json, os, shutil, sys, time
sys.path.insert(0, "src")
from jevbench.budget import Ledger
from jevbench.runner import Runner
from jevbench.summarize import summarize
from jevbench.tasks import load_jsonl
from ryotide.jevbench_adapter import MlxJevLocalAdapter

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", default="vendor/jevbench/datasets/public/original.jsonl,"
                                   "vendor/jevbench/datasets/public/easy.jsonl,"
                                   "vendor/jevbench/datasets/public/hard.jsonl")
ap.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
ap.add_argument("--orders", type=int, default=2)
ap.add_argument("--no-chat", action="store_true")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--tag", default="run")
ap.add_argument("--franken", default=None, help="delta,shift -> duplicate query heads")
ap.add_argument("--dtype", default=None, help="float32 | bfloat16")
ap.add_argument("--idk", action="store_true", help="offer an unscored 'cannot determine' option")
ap.add_argument("--idk-first", action="store_true", help="place the escape hatch at slot A")
ap.add_argument("--repeat", type=int, default=1, help="emit the question body N times (echo trick)")
ap.add_argument("--prefix", default=None, help="pin the answer prefix (skips calibration)")
ap.add_argument("--backend", choices=("mlx", "torch"), default="mlx")
ap.add_argument("--device", default=None, help="torch only: cuda | mps | cpu")
ap.add_argument("--revision", default=None, help="pin the model revision (torch)")
ap.add_argument("--qfirst", action="store_true", help="also emit the question before the state")
ap.add_argument("--instruction", default=None, help="override the closing instruction line")
ap.add_argument("--marker", default=None, help="pin the marker surface form, e.g. '{}' or ' {}'")
a = ap.parse_args()

tasks = []
for part in a.tasks.split(","):
    tasks.extend(load_jsonl(part.strip()))
if a.limit:
    tasks = tasks[:a.limit]

out = f"results/jevbench/{a.tag}"
shutil.rmtree(out, ignore_errors=True); os.makedirs(out, exist_ok=True)
raw = f"/tmp/jevbench-raw/{a.tag}"; shutil.rmtree(raw, ignore_errors=True)

fr = tuple(float(x) for x in a.franken.split(",")) if a.franken else None
if fr: fr = (fr[0], int(fr[1]))
ad = MlxJevLocalAdapter(endpoint=a.model, orders=a.orders, chat=not a.no_chat,
                        franken=fr, dtype=a.dtype, idk=a.idk,
                        idk_first=a.idk_first, repeat=a.repeat,
                        pin_prefix=a.prefix, pin_marker=a.marker,
                        instruction=a.instruction, question_first=a.qfirst,
                        backend=a.backend, device=a.device, revision=a.revision)
t0 = time.perf_counter(); ad.load()
print(f"[warm load {time.perf_counter()-t0:.1f}s] model={a.model} orders={a.orders} "
      f"chat={not a.no_chat} franken={fr} dtype={a.dtype} tasks={len(tasks)}", flush=True)

# Calibrate the read position once, on a spread of real tasks, before the clock
# starts. Which prefix a model wants is model-specific (Qwen2.5 "Answer:",
# Qwen3/Gemma "Answer: **", Qwen3.5 none) and a single task picks it unreliably.
import random as _r
_pool = [tasks[i] for i in _r.Random(0).sample(range(len(tasks)), min(6, len(tasks)))]
if a.prefix is None:
    ad.calibrate(_pool)
else:
    print(f"[pinned] prefix={a.prefix!r} marker={a.marker or '{}'!r}", flush=True)

runner = Runner(ad, Ledger(f"{out}/ledger.jsonl", cap_usd=1000.0), raw_dir=raw,
                default_reserve_usd=0.0)  # local weights: no provider tariff to reserve against
records = runner.run_all(tasks, progress_every=25, results_path=f"{out}/results.jsonl")
s = summarize(tasks, records)
json.dump(s, open(f"{out}/summary.json", "w"), indent=1, default=str)

ok = [r for r in records if r["ok"]]
acc = sum(1 for r in records if r["correct"]) / max(len(records), 1)
lat = sorted(r["latency_s"] for r in ok)
print(f"\n=== {a.tag} ===")
print(f"  n={len(records)}  ok={len(ok)}  valid={sum(1 for r in records if r['valid'])}")
print(f"  accuracy {acc:.3f}")
if lat:
    print(f"  latency p50 {lat[len(lat)//2]*1000:.0f}ms  p95 {lat[int(len(lat)*.95)-1]*1000:.0f}ms")
fam = {}
for r in records:
    d = fam.setdefault(r["family"], [0, 0]); d[1] += 1; d[0] += bool(r["correct"])
for k, (c, n) in sorted(fam.items()):
    print(f"    {k:10s} {c}/{n} = {c/n:.3f}")
print(f"  -> {out}/summary.json")
