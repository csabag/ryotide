"""GLUE through the CURRENT local pipeline: calibrated read position, conditional
question repetition, masked single-forward-pass logits.

GLUE items are wrapped as JevBench-shaped tasks (state + instructions + per-label
criteria) so our models see exactly the structure they were tuned on, and the
same 150 items per task that the Decisions API saw (same seed, same shuffle).
"""
import argparse, json, random, sys, time
from dataclasses import dataclass, field
sys.path.insert(0, "src")
from ryotide.jevbench_adapter import MlxJevLocalAdapter

@dataclass
class T:                       # minimal stand-in for a jevbench Task
    id: str; state: str; question: dict; labels: list; expected: str
    family: str = "glue"; split: str = "public"; group: str = None
    provenance: dict = field(default_factory=dict)

SPEC = {
 "sst2": (lambda e: f'Review: "{e["sentence"].strip()}"',
   "Is the sentiment of this review positive or negative?",
   ["positive","negative"], {"positive":"The review expresses positive sentiment.",
                             "negative":"The review expresses negative sentiment."},
   lambda lab: "positive" if lab==1 else "negative"),
 "rte": (lambda e: f'Premise: "{e["sentence1"].strip()}"\nHypothesis: "{e["sentence2"].strip()}"',
   "Does the premise entail the hypothesis?",
   ["yes","no"], {"yes":"The hypothesis follows from the premise.",
                  "no":"The hypothesis does not follow from the premise."},
   lambda lab: "yes" if lab==0 else "no"),
 "mrpc": (lambda e: f'Sentence 1: "{e["sentence1"].strip()}"\nSentence 2: "{e["sentence2"].strip()}"',
   "Do these two sentences mean the same thing?",
   ["yes","no"], {"yes":"The sentences are semantically equivalent.",
                  "no":"The sentences differ in meaning."},
   lambda lab: "yes" if lab==1 else "no"),
 "qnli": (lambda e: f'Question: {e["question"].strip()}\nSentence: "{e["sentence"].strip()}"',
   "Does the sentence contain the answer to the question?",
   ["yes","no"], {"yes":"The sentence answers the question.",
                  "no":"The sentence does not answer the question."},
   lambda lab: "yes" if lab==0 else "no"),
 "cola": (lambda e: f'Sentence: "{e["sentence"].strip()}"',
   "Is this sentence grammatically acceptable English?",
   ["yes","no"], {"yes":"Grammatically acceptable.","no":"Not grammatically acceptable."},
   lambda lab: "yes" if lab==1 else "no"),
}
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--n", type=int, default=150)
ap.add_argument("--repeat", type=int, default=2); ap.add_argument("--tag", required=True)
ap.add_argument("--prefix", default=None); ap.add_argument("--marker", default="{}")
a = ap.parse_args()
from datasets import load_dataset

ad = MlxJevLocalAdapter(endpoint=a.model, orders=1, repeat=a.repeat,
                        pin_prefix=a.prefix, pin_marker=a.marker); ad.load()
out, pool = {}, []
built = {}
for name,(state_of,instr,labels,crit,exp_of) in SPEC.items():
    ds = load_dataset("nyu-mll/glue", name, split="validation")
    idx = list(range(len(ds))); random.Random(0).shuffle(idx); idx = idx[:a.n]
    ts=[T(id=f"{name}-{i}", state=state_of(ds[i]),
          question={"type":"noul" if labels==["yes","no"] else "choice",
                    "instructions":instr,"criteria":crit},
          labels=labels, expected=exp_of(int(ds[i]["label"]))) for i in idx]
    built[name]=ts; pool += ts[:2]
if a.prefix is None: ad.calibrate(pool[:6])
print(f"[{a.tag}] prefix={ad._suffix!r} marker={ad._pattern!r} mass={ad._suffix_mass:.4f}", flush=True)
for name, ts in built.items():
    ok=n=0; t0=time.time(); masses=[]
    for t in ts:
        r = ad.run(t)
        if not r.ok: continue
        ok += (max(r.probs, key=r.probs.get) == t.expected); n += 1
        masses.append((r.raw or {}).get("runtime",{}).get("marker_mass",0))
    maj = max(sum(1 for t in ts if t.expected==l) for l in ts[0].labels)/len(ts)
    out[name]={"n":n,"accuracy":round(ok/max(n,1),4),"majority":round(maj,4),
               "marker_mass":round(sum(masses)/max(len(masses),1),4),
               "s":round(time.time()-t0,1)}
    print(f"  {name:5s} acc {out[name]['accuracy']:.3f}  (majority {maj:.3f})  "
          f"mass {out[name]['marker_mass']:.3f}  [{out[name]['s']}s]", flush=True)
json.dump(out, open(f"results/glue_{a.tag}.json","w"), indent=1)
