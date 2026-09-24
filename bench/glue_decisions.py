"""Run GLUE through the Decisions API, and through our local classifier, on the
SAME items -- so Jev and our models are compared like for like.

Every GLUE task here is binary, so each maps onto Jev's native `noul` type:
a probability that the "true" criterion holds. Our own models answer the same
item through the ordinary masked-logit path.
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, "src")
from jevbench.adapters.base import http_post_json

for line in open(".env"):
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

# task -> (hf config, state builder, instructions, {true:..., false:...}, label for expected==1)
TASKS = {
 "sst2": ("sst2", lambda e: f'Review: "{e["sentence"].strip()}"',
    "Is the sentiment of this review positive?",
    {"true": "The review expresses positive sentiment.",
     "false": "The review expresses negative sentiment."}, 1),
 "rte": ("rte", lambda e: f'Premise: "{e["sentence1"].strip()}"\nHypothesis: "{e["sentence2"].strip()}"',
    "Does the premise entail the hypothesis?",
    {"true": "The hypothesis follows from the premise.",
     "false": "The hypothesis does not follow from the premise."}, 0),
 "mrpc": ("mrpc", lambda e: f'Sentence 1: "{e["sentence1"].strip()}"\nSentence 2: "{e["sentence2"].strip()}"',
    "Do these two sentences mean the same thing?",
    {"true": "The sentences are semantically equivalent.",
     "false": "The sentences differ in meaning."}, 1),
 "qnli": ("qnli", lambda e: f'Question: {e["question"].strip()}\nSentence: "{e["sentence"].strip()}"',
    "Does the sentence contain the answer to the question?",
    {"true": "The sentence answers the question.",
     "false": "The sentence does not answer the question."}, 0),
 "cola": ("cola", lambda e: f'Sentence: "{e["sentence"].strip()}"',
    "Is this sentence grammatically acceptable English?",
    {"true": "Grammatically acceptable.",
     "false": "Not grammatically acceptable."}, 1),
}
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--model", default="typesafe/jev-1.13")
ap.add_argument("--tasks", nargs="+", default=list(TASKS))
ap.add_argument("--out", default="results/glue_jev.json")
a = ap.parse_args()
KEY = os.environ["OPENROUTER_API_KEY"]
from datasets import load_dataset

out, total_cost = {}, 0.0
for name in a.tasks:
    cfg, state_of, instr, crit, true_label = TASKS[name]
    ds = load_dataset("nyu-mll/glue", cfg, split="validation")
    idx = list(range(len(ds))); random.Random(0).shuffle(idx); idx = idx[:a.n]
    ok = n = 0; probs = []; lat = []; cost = 0.0
    t0 = time.time()
    for i in idx:
        e = ds[i]
        body = {"model": a.model, "state": state_of(e),
                "questions": {"d": {"type": "noul", "instructions": instr, "criteria": crit}}}
        try:
            st, resp, dt = http_post_json("https://openrouter.ai/api/v1/systemone", body,
                {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}, 60)
        except Exception as ex:
            print(f"  {name}: {type(ex).__name__}"); continue
        if st != 200 or not isinstance(resp, dict) or "answers" not in resp:
            print(f"  {name}: HTTP {st} {str(resp)[:120]}"); continue
        p = resp["answers"]["d"]["noul"]
        pred = true_label if p >= 0.5 else (1 - true_label)
        ok += (pred == int(e["label"])); n += 1
        probs.append(p); lat.append(dt); cost += (resp.get("usage") or {}).get("cost", 0) or 0
    maj = max(sum(1 for i in idx if int(ds[i]["label"]) == c) for c in (0, 1)) / len(idx)
    out[name] = {"n": n, "accuracy": round(ok / max(n,1), 4), "majority": round(maj, 4),
                 "mean_p": round(sum(probs)/max(len(probs),1), 3),
                 "p50_ms": round(sorted(lat)[len(lat)//2]*1000) if lat else None,
                 "cost_usd": round(cost, 5)}
    total_cost += cost
    print(f"  {name:5s} acc {out[name]['accuracy']:.3f}  (majority {maj:.3f})  "
          f"n={n}  p50 {out[name]['p50_ms']}ms  ${cost:.4f}  [{time.time()-t0:.0f}s]", flush=True)
json.dump(out, open(a.out, "w"), indent=1)
print(f"\ntotal cost ${total_cost:.4f}   -> {a.out}")
