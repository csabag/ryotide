"""Thin HTTP transport for library-only entrants (SemIf, open-alternative-jev), run on the pod.

    python r2_serve.py <system-key> <port>

Loads the system's in-process adapter once (weights loaded before serving), then answers
POST /run {"task": <canonical record>} with the adapter's result. No mapping happens
here that the in-process run does not do: the same adapter code runs server-side, so the
remote run differs from the in-process run only by the network hop the client pays.
"""
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B / "repo"))
from jevbench.adapters import SemIfDirectAdapter, SgSystemOneAdapter, So1DeciderAdapter  # noqa: E402
from jevbench.tasks import Task  # noqa: E402

PLAN = {s["key"]: s for s in json.loads((B / "r2_plan.json").read_text())["systems"]}
key, port = sys.argv[1], int(sys.argv[2])
cfg = PLAN[key]
kind = {"semif_direct": SemIfDirectAdapter, "so1_decider": So1DeciderAdapter,
        "sg_system_one": SgSystemOneAdapter}[cfg["adapter"]]
ADAPTER = kind(endpoint=cfg["endpoint"], model=cfg["model"], revision=cfg.get("revision"))
if cfg.get("request_options"):
    ADAPTER.request_options = cfg["request_options"]
t0 = time.perf_counter(); ADAPTER.load(); print(f"[serve] {key} loaded in {time.perf_counter()-t0:.1f}s", flush=True)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        r = ADAPTER.run(Task.from_dict(body["task"]))
        out = json.dumps({**r.to_public(), "raw": r.raw}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)


ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
