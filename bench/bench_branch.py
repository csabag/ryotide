"""Spec step 4 measurement: does branching actually cost anything?

Answers three questions the spec leaves open:
  1. KV memory for N branches at a realistic state size (flagged "unquantified")
  2. wall-clock cost of fork vs trim vs lazy as N grows
  3. what fraction of a classification is the shared prefill vs the question --
     which is what decides whether masked batched attention is worth building
"""
import argparse, json, sys, time
sys.path.insert(0, "src")
import mlx.core as mx
from ryotide.model import load_model, describe
from ryotide.classify import Classifier, LabelSet
from ryotide.branch import (BranchedClassifier, cache_bytes, fork_cache,
                            projected_branch_memory)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
ap.add_argument("--state-tokens", type=int, default=6000)
ap.add_argument("--ns", type=int, nargs="+", default=[1, 5, 10, 25, 50])
ap.add_argument("--out", default="results/branch.json")
a = ap.parse_args()

model, tok = load_model(a.model)
clf = Classifier(model, tok)
info = describe(model, a.model)
labels = LabelSet.resolve(tok, ["yes", "no"])

# A state of roughly the right shape and size: a log of facts to interrogate.
para = ("Incident 4471: the ingest worker stalled at 02:14 UTC after the upstream "
        "queue backed up past 80k messages. Retries were exhausted; the on-call "
        "engineer restarted the pool and throughput recovered by 02:41. No data loss "
        "was observed. Follow-up: raise the queue alarm threshold and add a watchdog. ")
state_text = ""
while len(clf.encode(state_text)) < a.state_tokens:
    state_text += para
state_tokens = clf.encode(state_text)[: a.state_tokens]
state_text = tok.decode(state_tokens)

QUESTIONS = [
    "\nQ: Did the incident cause data loss? Answer yes or no.\nA:",
    "\nQ: Was the queue involved? Answer yes or no.\nA:",
    "\nQ: Did an engineer intervene? Answer yes or no.\nA:",
    "\nQ: Is a follow-up action recorded? Answer yes or no.\nA:",
    "\nQ: Did the incident happen during business hours? Answer yes or no.\nA:",
]

bc = BranchedClassifier(clf)
mx.reset_peak_memory()
t0 = time.time()
state_len = bc.set_state(state_text)
mx.eval([c.state for c in bc.state_cache])
prefill_s = time.time() - t0
state_kv = cache_bytes(bc.state_cache)

print(f"model={a.model}")
print(f"state: {state_len} tokens, prefill {prefill_s*1000:.0f}ms "
      f"({state_len/prefill_s:.0f} tok/s), KV {state_kv/1e6:.1f}MB "
      f"({state_kv/state_len:.0f} B/tok, predicted {info.kv_bytes_per_token})")

# --- single-question marginal cost ---------------------------------------
q = QUESTIONS[0]
qt = clf.encode(q)
for _ in range(2):   # warm
    b = fork_cache(bc.state_cache); clf.classify(b, qt, labels)
t0 = time.time()
REPS = 10
for _ in range(REPS):
    b = fork_cache(bc.state_cache)
    d = clf.classify(b, qt, labels)
    del b
per_q = (time.time() - t0) / REPS
print(f"one question on a warm state: {per_q*1000:.1f}ms  "
      f"({len(qt)} question tokens) -> {prefill_s/per_q:.0f}x cheaper than re-prefilling")

rows = []
for n in a.ns:
    qs = [QUESTIONS[i % len(QUESTIONS)] for i in range(n)]
    row = {"n": n, "state_len": state_len}
    ref = None
    for strat in ("trim", "fork", "lazy"):
        mx.reset_peak_memory()
        base_active = mx.get_active_memory()
        t0 = time.time()
        res = bc.classify_many(qs, labels, strategy=strat)
        if strat == "lazy":                       # + materialise one branch to generate from
            gen_branch = bc.materialize(qs[0])
        mx.eval([])
        el = time.time() - t0
        peak = mx.get_peak_memory() - base_active
        live_kv = sum(cache_bytes(r.cache) for r in res if r.cache is not None)
        if strat == "lazy":
            live_kv += cache_bytes(gen_branch); del gen_branch
        labs = [r.decision.label for r in res]
        if ref is None:
            ref = labs
        row[strat] = {
            "seconds": round(el, 4),
            "ms_per_question": round(1000 * el / n, 2),
            "peak_alloc_mb": round(peak / 1e6, 1),
            "retained_branch_kv_mb": round(live_kv / 1e6, 1),
            "agrees_with_trim": labs == ref,
        }
        for r in res:
            r.cache = None
    print(f"N={n:3d}  " + "  ".join(
        f"{s}: {row[s]['ms_per_question']:6.1f}ms/q retained={row[s]['retained_branch_kv_mb']:7.1f}MB"
        f"{'' if row[s]['agrees_with_trim'] else ' MISMATCH!'}"
        for s in ("trim", "fork", "lazy")))
    rows.append(row)

out = {
    "hardware": "Mac mini Mac16,10 base M4, 10-core CPU (4P+6E), 24GB",
    "model": a.model,
    "kv_bytes_per_token": info.kv_bytes_per_token,
    "state_len": state_len,
    "state_prefill_seconds": round(prefill_s, 4),
    "state_prefill_tok_per_s": round(state_len / prefill_s, 1),
    "state_kv_mb": round(state_kv / 1e6, 2),
    "marginal_question_ms": round(per_q * 1000, 2),
    "projection_at_6k": {n: projected_branch_memory(info.kv_bytes_per_token, 6000, n)
                         for n in a.ns},
    "rows": rows,
}
json.dump(out, open(a.out, "w"), indent=1)
print(f"\nwrote {a.out}")
