"""Run one round-2 system on one v1.1 tier, ON the pod (client on the same machine).
    python3 r2_run.py <key> easy|v1 [--limit N] [--suffix S]
Same harness, same tiers and same rules as v1.1 (serial, no retries, immutable run folders)."""
import argparse, json, os, subprocess, sys
from pathlib import Path
B = Path(__file__).resolve().parent
PLAN = json.loads((B / "r2_plan.json").read_text())
TIERS = {"easy": "datasets/public/easy.jsonl,../private/easy-heldout.jsonl",
         "hard": "/home/flori/jobs/jev-class-benchmark-20260919/repo/datasets/public/hard.jsonl,/home/flori/jobs/jev-class-benchmark-20260919/private/hard-heldout.jsonl",
         "v1": "datasets/public/original.jsonl,../private/heldout.jsonl,../private/imported/router.jsonl,../private/imported/judge.jsonl"}
ap = argparse.ArgumentParser(); ap.add_argument("key"); ap.add_argument("tier", choices=TIERS)
ap.add_argument("--limit", type=int); ap.add_argument("--suffix", default="")
ap.add_argument("--remote", help="run from this machine against the pod at this base URL (network included)")
a = ap.parse_args()
cfg = dict(next(s for s in PLAN["systems"] if s["key"] == a.key))
if a.remote:
    cfg["python"] = sys.executable
    if cfg["adapter"] in ("semif_direct", "so1_decider", "sg_system_one"):
        cfg["adapter"] = "remote_inproc"
    cfg["endpoint"] = a.remote
    cfg["measured_from"] = "Sandy (Hetzner, Germany), over the internet to the pod"
out = B / ("runs-net" if a.remote else "runs") / f"{a.key}--{a.tier}{a.suffix}"
if out.exists(): sys.exit(f"{out} exists; runs are immutable")
out.mkdir(parents=True)
cmd = [cfg.get("python", sys.executable), "-m", "jevbench.cli", "run", "--tasks", TIERS[a.tier],
       "--adapter", cfg["adapter"], "--model", cfg["model"], "--key-env", "",
       "--results", str(out / "results.jsonl"), "--raw-dir", str(out / "raw"),
       "--ledger", str(out.parent / "ledger.jsonl"), "--cap-usd", "0.01", "--reserve-usd", "0",
       "--delay-s", "0", "--run-label", f"{a.key}/{a.tier}", "--manifest", str(out / "manifest.json"),
       "--cost-basis", cfg["cost_basis"]]
if cfg.get("endpoint"): cmd += ["--endpoint", cfg["endpoint"]]
if cfg.get("revision") and cfg["adapter"] != "remote_inproc": cmd += ["--revision", cfg["revision"]]
if cfg.get("request_options"): cmd += ["--request-options", json.dumps(cfg["request_options"])]
if a.limit: cmd += ["--limit", str(a.limit)]
env = dict(os.environ, JEVBENCH_WARM_LOAD="1", OPENAI_API_KEY="", PYTHONPATH=str(B / "repo") + ":" + os.environ.get("PYTHONPATH", ""))
(out / "system.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))
print("[r2]", a.key, a.tier, "->", out, flush=True)
rc = subprocess.run(cmd, cwd=str(B / "repo"), env=env).returncode
print("[r2] exit", rc, flush=True); sys.exit(rc)
