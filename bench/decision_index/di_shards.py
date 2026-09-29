"""Run the Decision Index suite split across N servers, then merge and score once.

The kit runs one request at a time; our server answers one at a time too. To use a big GPU,
start N servers and run shard K of N against each: a request belongs to shard
crc32(run_id) % N, so shards are disjoint, cover the edition exactly, and mix long and short
requests evenly. Each shard is an ordinary `decision_index.runner.run` (same code path,
resume, status files); `merge` concatenates the shard results and scores them with the kit.

    python di_shards.py run   --shard 0 --of 3 --port 8001 --model ryotide-qwen --out runs/qwen4b/shard0
    # re-split: stop a model's runners, then run --of M over what they left (errors are retried)
    python di_shards.py run --shard 0 --of 4 --port 8013 --model ryotide-model --out runs/model/b0 --skip-done runs/model/shard0 runs/model/shard1
    python di_shards.py merge --out runs/qwen4b runs/qwen4b/shard0 runs/qwen4b/shard1 runs/qwen4b/shard2
"""
import argparse
import json
import zlib
from pathlib import Path

from decision_index import editions
from decision_index.pipeline import score_run
from decision_index.runner import run
from decision_index.suite.io import Suite, atomic_json, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=("run", "merge"))
ap.add_argument("shards", nargs="*")
ap.add_argument("--suite-dir", default="suite-0.2")
ap.add_argument("--edition", default=editions.DEFAULT)
ap.add_argument("--shard", type=int)
ap.add_argument("--of", type=int)
ap.add_argument("--port", type=int)
ap.add_argument("--model", default="ryotide")
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int)
ap.add_argument("--skip-done", nargs="*", default=[], help="run: shard dirs whose non-error results are kept (re-split a running model)")
a = ap.parse_intermixed_args()
suite = Suite(Path(a.suite_dir), a.edition)

if a.cmd == "run":
    corpus = suite.verify()["sha256"]
    done = {r["run_id"] for d in a.skip_done for r in read_jsonl(Path(d) / "results.jsonl", complete_lines_only=True)
            if r["status"] != "error"}
    keep = lambda e: suite.in_edition(e) and e["run_id"] not in done and zlib.crc32(e["run_id"].encode()) % a.of == a.shard
    opts = {"base_url": f"http://127.0.0.1:{a.port}", "model": a.model}
    print(json.dumps(run("http", opts, suite.row_paths, Path(a.out), limit=a.limit, corpus_sha256=corpus, keep=keep)))
else:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seen = {}
    for s in a.shards:                       # last line per run_id wins (a resumed error retried)
        for r in read_jsonl(Path(s) / "results.jsonl", complete_lines_only=True):
            seen[r["run_id"]] = r
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in seen.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    env = json.loads((Path(a.shards[0]) / "environment.json").read_text())
    env["shards"] = {"count": len(a.shards), "rule": "crc32(run_id) % count", "note": "one server per shard on the same GPU; latency here is under contention, not the board's single-process measurement",
                     "environments": [json.loads((Path(s) / "environment.json").read_text()) for s in a.shards]}
    atomic_json(out / "environment.json", env)
    scores = score_run(suite, out / "results.jsonl", "http", out)
    print(json.dumps({k: scores.get(k) for k in ("edition", "decision_index", "completed", "complete")}, indent=1))
