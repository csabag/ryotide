"""Batched vs sequential multi-question reads (torch): same answers, how much faster?

Runs the same /v1/systemone-style requests through the engine twice -- RYOTIDE_NO_BATCH=1
(one forward pass per question, the old path) and batched (TorchClassifier.read_batch) --
and compares every question's probabilities and chosen label, and the wall time.

Requests: multi-question cases from --data files (optional), a 30-question retrieval
request with long candidates (memory budgets), and a request mixing a 40-option question
(two-token codes, stays sequential) with ordinary ones.

    uv run --extra torch python bench/probes/batch_check.py --model Qwen/Qwen3.5-4B \\
        --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a [--device mps] [--limit 10]
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "vendor/jevbench")
from ryotide.jevbench_adapter import MlxJevLocalAdapter  # noqa: E402
from ryotide.server import tasks_from_request  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
ap.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--device", default=None)
ap.add_argument("--data", action="append", help="JSONL of multi-question cases ({id, state, questions}); repeatable")
ap.add_argument("--limit", type=int, default=10, help="cases per --data file")
ap.add_argument("--dtype", default=None, help="e.g. float32 to separate arithmetic noise from logic errors")
ap.add_argument("--only", default=None, help="run only requests whose name contains this")
ap.add_argument("--retrieval-n", type=int, default=30)
a = ap.parse_args()

ad = MlxJevLocalAdapter(endpoint=a.model, orders=1, repeat=2, pin_prefix="Answer: **", pin_marker="{}",
                        backend="torch", device=a.device, revision=a.revision, temperature=a.temperature, dtype=a.dtype)
ad.load()

requests = []
for path in a.data or []:
    if os.path.exists(path):
        for line in list(open(path, encoding="utf-8"))[: a.limit]:
            r = json.loads(line)
            requests.append((f"demo {r['id']}", {"state": r["state"], "questions": r["questions"]}))
rng = random.Random(0)
words = ("the service account returns a list of records for the given user query and date range "
         "with optional filters for status region and owner sorted by creation time").split()
doc = lambda n: " ".join(rng.choice(words) for _ in range(n))
YN = {"yes": "Relevant to the query.", "no": "Not relevant to the query."}
requests.append((f"retrieval {a.retrieval_n} x long candidates", {"state": {"query": "Find the tool that lists invoices for a customer."},
                 "questions": {f"c{i:02d}": {"type": "choice", "criteria": YN,
                               "instructions": {"task": "Is this candidate tool relevant to the query?", "candidate": doc(rng.randint(60, 400))}}
                               for i in range(a.retrieval_n)}}))
requests.append(("mixed: 40 options (codes) + ordinary", {"state": "Customer writes: my parcel arrived damaged and I want a refund.",
                 "questions": {"intent": {"type": "choice", "instructions": "Which intent?", "criteria": {f"intent_{i:02d}": "" for i in range(40)}},
                               "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
                               "tone": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "angry"]},
                               "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "", "logistics": "", "tech": ""}}}}))


def run(body, batched):
    os.environ["RYOTIDE_NO_BATCH"] = "0" if batched else "1"
    tasks = tasks_from_request(body)
    t0 = time.perf_counter()
    res = ad.run_many(tasks) if len(tasks) > 1 else [ad.run(tasks[0])]
    dt = time.perf_counter() - t0
    bad = [r.error for r in res if not r.ok]
    if bad:
        raise SystemExit(f"decision failed: {bad[0]}")
    return {t.key: r.probs for t, r in zip(tasks, res)}, dt


if a.only:
    requests = [r for r in requests if a.only in r[0]]
run(requests[-1][1], True); run(requests[-1][1], False)        # warm-up both paths
worst, flips, t_seq, t_bat, n_q = 0.0, 0, 0.0, 0.0, 0
for name, body in requests:
    seq, ts = run(body, False)
    bat, tb = run(body, True)
    t_seq += ts; t_bat += tb; n_q += len(seq)
    d = max(abs(seq[k][lab] - bat[k][lab]) for k in seq for lab in seq[k])
    f = sum(max(seq[k], key=seq[k].get) != max(bat[k], key=bat[k].get) for k in seq)
    worst = max(worst, d); flips += f
    print(f"{name:40s} {len(seq):3d} q  sequential {ts*1000:7.0f} ms  batched {tb*1000:7.0f} ms  "
          f"x{ts/max(tb,1e-9):4.1f}  max|dp| {d:.2e}  label flips {f}")
print(f"\n{len(requests)} requests, {n_q} questions: sequential {t_seq:.1f} s, batched {t_bat:.1f} s "
      f"(x{t_seq/max(t_bat,1e-9):.1f}); max |dp| {worst:.2e}; label flips {flips}")
print("PASS" if flips == 0 and worst < 0.02 else "CHECK: differences above tolerance")
