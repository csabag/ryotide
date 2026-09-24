"""TypeSafe-compatible HTTP server for RYOTIDE.

Serves the wire format JevBench's stock `typesafe` adapter speaks, so the
benchmark can drive RYOTIDE without any RYOTIDE-specific code:

  POST /v1/systemone   {"state": ..., "model": ..., "questions": {"decision": q}}
                       q = {"type": "noul"|"choice"|"score", "instructions": str,
                            "criteria": dict | list}
  ->  {"model": ..., "answers": {"decision": {...}}, "usage": {...}}
  GET  /health         model, engine, decision config and a prompt hash

Labels are rebuilt from the question alone, because the wire never carries them:
noul -> ["no", "yes"]; score -> "0".."n-1" over the criteria list (a list, so its
order is the author's); choice -> the criteria keys. A JSON object's key order is
not meaningful, so by default choice options are put in a canonical NATURAL order:
runs of digits compare as numbers ("6_credits" < "9_credits" < "12_credits"), all
else alphabetically. Numeric scales then read as scales, and the decision no longer
depends on how a client happened to order its keys. `--option-order received`
keeps the request's order instead.

    PYTHONPATH=src python -m ryotide.server --model mlx-community/gemma-4-e4b-it-8bit
    PYTHONPATH=src python -m ryotide.server --backend torch --model google/gemma-4-E4B-it \
        --revision ee0ef6023621cff504d758262d4e04895a5af4a2

No authentication: bind to loopback (the default) unless something in front of it
provides access control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

try:
    import jevbench  # noqa: F401  (the adapter returns the harness's result type)
except ImportError:
    _vendor = os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "jevbench")
    sys.path.insert(0, os.path.abspath(_vendor))

from .jevbench_adapter import DEFAULT_INSTRUCTION, MlxJevLocalAdapter

NOUL_LABELS = ["no", "yes"]

# One pinned configuration per benchmark entry. The temperature is fit on the
# synthetic typed-decision set (never on JevBench items) with
# bench/fit_temperature.py; see docs/SUBMISSION.md.
PRESETS = {
    "ryotide-gemma": dict(backend="torch", model="google/gemma-4-E4B-it",
                          revision="ee0ef6023621cff504d758262d4e04895a5af4a2", temperature=1.0),
    "ryotide-qwen": dict(backend="torch", model="Qwen/Qwen3.5-4B",
                         revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", temperature=1.0),
}


def natural_key(label: str) -> list:
    """'12_credits' -> [(0, 12), (1, '_credits')]: digits compare as numbers."""
    return [(0, int(t)) if t.isdigit() else (1, t.lower())
            for t in re.split(r"(\d+)", label) if t]


class BadRequest(ValueError):
    pass


def task_from_request(body: dict, option_order: str = "natural") -> SimpleNamespace:
    """Rebuild the adapter's task view from one /v1/systemone request."""
    if not isinstance(body, dict) or "state" not in body:
        raise BadRequest("body must be an object with 'state' and 'questions'")
    qs = body.get("questions")
    if not isinstance(qs, dict) or len(qs) != 1:
        raise BadRequest("exactly one question is supported per request")
    (key, q), = qs.items()
    if not isinstance(q, dict) or not isinstance(q.get("instructions"), str):
        raise BadRequest(f"question {key!r} needs 'type' and 'instructions'")
    qtype, crit = q.get("type"), q.get("criteria")
    if qtype == "noul":
        labels = list(NOUL_LABELS)
    elif qtype == "choice":
        if not isinstance(crit, dict) or len(crit) < 2:
            raise BadRequest("choice needs a 'criteria' object with at least two options")
        labels = [str(k) for k in crit]
        if option_order == "natural":
            labels.sort(key=natural_key)
    elif qtype == "score":
        if not isinstance(crit, list) or len(crit) < 2:
            raise BadRequest("score needs a 'criteria' list with at least two levels")
        labels = [str(i) for i in range(len(crit))]
    else:
        raise BadRequest(f"unsupported question type {qtype!r}")
    question = {"type": qtype, "instructions": q["instructions"]}
    if crit is not None:
        question["criteria"] = crit
    return SimpleNamespace(key=key, state=body["state"], question=question, labels=labels,
                           task_id=f"wire-{key}", family=None)


def answer_for(qtype: str, probs: dict) -> dict:
    top = max(probs, key=probs.get)
    if qtype == "noul":
        return {"type": "noul", "noul": float(probs["yes"])}
    if qtype == "choice":
        return {"type": "choice", "choice": top, "probabilities": probs}
    return {"type": "score", "score": int(top), "probabilities": probs}


class Engine:
    def __init__(self, a: argparse.Namespace):
        self.adapter = MlxJevLocalAdapter(
            endpoint=a.model, orders=a.orders, repeat=a.repeat,
            pin_prefix=a.prefix, pin_marker=a.marker, backend=a.backend,
            device=a.device, revision=a.revision, quant=a.quant, low_vram=a.low_vram,
            temperature=a.temperature)
        self.preset = a.preset
        self.backend = a.backend
        self.revision = a.revision
        self.model = a.model
        self.option_order = a.option_order
        self.lock = threading.Lock()
        self.config = {"orders": a.orders, "repeat": a.repeat, "answer_prefix": a.prefix,
                       "marker_pattern": a.marker, "instruction": DEFAULT_INSTRUCTION,
                       "layout": "state / question / question (echo always)",
                       "temperature": a.temperature, "option_order": a.option_order}
        self.prompt_hash = hashlib.sha256(json.dumps(self.config, sort_keys=True)
                                          .encode()).hexdigest()[:12]

    def decide(self, body: dict) -> dict:
        task = task_from_request(body, self.option_order)
        with self.lock:                      # MLX is not thread-safe
            res = self.adapter.run(task)
        if not res.ok:
            raise RuntimeError(res.error or "decision failed")
        return {"model": self.model,
                "answers": {task.key: answer_for(task.question["type"], res.probs)},
                "usage": res.usage,
                "runtime": {"latency_s": round(res.latency_s or 0.0, 4),
                            "marker_mass": res.raw["runtime"]["marker_mass"]}}

    def health(self) -> dict:
        dev = getattr(self.adapter._clf, "device", None)
        return {"status": "ok", "entry": self.preset, "model": self.model, "revision": self.revision,
                "engine": self.backend if self.backend == "mlx" else f"torch-{dev}",
                "prompt_hash": self.prompt_hash, "config": self.config,
                "weights": {"quant": self.adapter.quant or "none",
                            "offloaded_to_cpu": getattr(self.adapter._clf, "offloaded", [])}}


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                self._send(200, engine.health())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                self._send(200, engine.decide(body))
            except (BadRequest, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:           # never leak a traceback to the client
                self._send(500, {"error": f"{type(e).__name__}: {str(e)[:200]}"})

        def log_message(self, fmt, *args):  # quiet: one line per request is noise
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="RYOTIDE TypeSafe-compatible server")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None,
                    help="a pinned benchmark entry; explicit flags below override it")
    ap.add_argument("--model", default=None)
    ap.add_argument("--backend", choices=("mlx", "torch"), default=None)
    ap.add_argument("--temperature", type=float, default=None,
                    help="softmax temperature over the option markers")
    ap.add_argument("--device", default=None, help="torch only: cuda | mps | cpu")
    ap.add_argument("--revision", default=None, help="pin the model revision")
    ap.add_argument("--quant", choices=("int8", "nf4"), default=None,
                    help="torch on CUDA: bitsandbytes weight quantization")
    ap.add_argument("--low-vram", action="store_true",
                    help="torch: keep embedding tables and unused towers on CPU")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8778)
    ap.add_argument("--orders", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--prefix", default="Answer: **")
    ap.add_argument("--marker", default="{}")
    ap.add_argument("--option-order", choices=("natural", "received"), default="natural",
                    help="choice options: canonical natural sort (default) or as received")
    a = ap.parse_args()
    base = PRESETS.get(a.preset, {}) if a.preset else {
        "backend": "mlx", "model": "mlx-community/gemma-4-e4b-it-8bit", "revision": None, "temperature": 1.0}
    for k, v in base.items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)

    engine = Engine(a)
    t0 = time.perf_counter()
    engine.adapter.load()
    engine.decide({"state": "The order shipped on Monday.",       # warm-up, never timed
                   "questions": {"decision": {"type": "noul",
                                              "instructions": "Has the order shipped?"}}})
    print(f"ryotide ready  model={a.model}  prompt_hash={engine.prompt_hash}  "
          f"load+warmup {time.perf_counter() - t0:.1f}s  http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(engine)).serve_forever()


if __name__ == "__main__":
    main()
