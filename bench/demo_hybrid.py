"""Spec step 5 end-to-end: classify N questions against one state, then continue
generating from the chosen branch with the label injected back into context.
Also A/Bs the json-vs-prose injection the spec flags as an untested guess."""
import json, sys, time; sys.path.insert(0,"src")
sys.argv = sys.argv
import mlx.core as mx
from ryotide.model import load_model, describe
from ryotide.classify import Classifier, LabelSet
from ryotide.branch import BranchedClassifier, cache_bytes
from ryotide.generate import classify_then_generate, ab_injection

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
model, tok = load_model(MODEL); clf = Classifier(model, tok)
labels = LabelSet.resolve(tok, ["yes","no"])

STATE = """Incident report 4471.
At 02:14 UTC the ingest worker pool stalled after the upstream queue backed up past
80,000 messages. Retry budget was exhausted within four minutes. The on-call engineer
was paged at 02:19, restarted the worker pool at 02:31, and throughput recovered fully
by 02:41. Checksums on the replay confirmed no data loss. Customer-facing latency was
elevated for 27 minutes but no requests were dropped. Follow-up actions: raise the
queue-depth alarm threshold from 50k to 30k, and add a watchdog that restarts stalled
workers automatically after 90 seconds.
"""
QUESTIONS = [
 "\nQ: Did this incident cause data loss? Answer yes or no.\nA:",
 "\nQ: Were any customer requests dropped? Answer yes or no.\nA:",
 "\nQ: Did a human have to intervene? Answer yes or no.\nA:",
 "\nQ: Are follow-up actions recorded? Answer yes or no.\nA:",
 "\nQ: Should this incident be escalated to a postmortem? Answer yes or no.\nA:",
]

bc = BranchedClassifier(clf)
t0=time.time(); n = bc.set_state(STATE); prefill=time.time()-t0
print(f"model={MODEL}\nstate={n} tokens, prefill {prefill*1000:.0f}ms, KV {cache_bytes(bc.state_cache)/1e6:.1f}MB\n")

t0=time.time(); res = bc.classify_many(QUESTIONS, labels, strategy="lazy"); el=time.time()-t0
print(f"--- classification ({len(QUESTIONS)} questions, {el*1000:.0f}ms total, {el*1000/len(QUESTIONS):.1f}ms each) ---")
for r in res:
    d=r.decision
    print(f"  {d.label:3s} p={d.probs[d.label]:.3f} margin={d.margin:.3f}  {r.question.strip()[3:70]}")

print("\n--- generation continuing from the escalation branch ---")
chosen = res[-1]
branch = bc.materialize(chosen.question)
g = classify_then_generate(clf, branch, chosen.decision,
      "\nIn two sentences, justify that recommendation to the on-call lead.\n",
      fmt="json", max_tokens=110)
print(f"[json injection] {g.n_tokens} tok, decode {g.decode_tps:.1f} tok/s, inject {g.inject_seconds*1000:.0f}ms")
print("  " + g.text.strip().replace("\n","\n  ")[:600])

print("\n--- A/B: injection format (spec flags this as an untested guess) ---")
ab = ab_injection(clf, lambda: bc.materialize(chosen.question), chosen.decision,
      "\nIn two sentences, justify that recommendation to the on-call lead.\n",
      max_tokens=96)
out={}
for fmt, g in ab.items():
    print(f"\n[{fmt}] {g.n_tokens} tok @ {g.decode_tps:.1f} tok/s")
    print("  " + g.text.strip().replace("\n","\n  ")[:420])
    out[fmt]={"tokens":g.n_tokens,"decode_tps":round(g.decode_tps,1),"text":g.text}
json.dump({"model":MODEL,"state_tokens":n,
  "classifications":[{"q":r.question.strip(),"label":r.decision.label,
                      "probs":r.decision.probs,"margin":r.decision.margin} for r in res],
  "injection_ab":out}, open("results/hybrid_demo.json","w"), indent=1)
print("\nwrote results/hybrid_demo.json")
