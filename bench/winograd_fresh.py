"""Five never-published Winograd twin pairs, written for this test.

Each schema is a sentence with two candidate antecedents and one ambiguous
pronoun. The twin changes a single discriminating phrase so the correct
antecedent flips; syntax is identical, so only world knowledge separates them.
Each twin is probed twice -- once asserting each antecedent -- giving balanced
labels and a consistency check: a model that pattern-matches will get the pair
but fail the twin.
"""
import argparse, json, os, sys, time
from dataclasses import dataclass, field
sys.path.insert(0, "src")

# (sentence, pronoun, entity_A, entity_B, correct) -- twins share A and B
SCHEMAS = [
 ("The archivist could not fit the ledger into the document scanner because it was too warped.",
  "it", "the ledger", "the document scanner", "the ledger"),
 ("The archivist could not fit the ledger into the document scanner because it was too narrow.",
  "it", "the ledger", "the document scanner", "the document scanner"),

 ("The mediator interrupted the claimant because she had been talking for twenty minutes.",
  "she", "the mediator", "the claimant", "the claimant"),
 ("The mediator interrupted the claimant because she needed to clarify the procedure.",
  "she", "the mediator", "the claimant", "the mediator"),

 ("The brewers moved the starter culture into the cold room because it was overactive.",
  "it", "the starter culture", "the cold room", "the starter culture"),
 ("The brewers moved the starter culture into the cold room because it had just been repaired.",
  "it", "the starter culture", "the cold room", "the cold room"),

 ("The luthier rejected the spruce top for the mandolin because it had a hidden crack.",
  "it", "the spruce top", "the mandolin", "the spruce top"),
 ("The luthier rejected the spruce top for the mandolin because it called for a darker tonewood.",
  "it", "the spruce top", "the mandolin", "the mandolin"),

 ("The registrar denied the request from the graduate because she had no record of the enrolment.",
  "she", "the registrar", "the graduate", "the registrar"),
 ("The registrar denied the request from the graduate because she had not paid the outstanding fee.",
  "she", "the registrar", "the graduate", "the graduate"),
]
ITEMS=[]
for sent, pro, a, b, correct in SCHEMAS:
    for cand in (a, b):
        ITEMS.append({"sentence": sent, "pronoun": pro, "candidate": cand,
                      "label": 1 if cand == correct else 0,
                      "schema": sent.split(" because ")[0]})
INSTR = "Does the pronoun refer to the stated candidate?"
CRIT  = {"true": "The pronoun refers to the stated candidate.",
         "false": "The pronoun refers to the other entity."}
state_of = lambda it: (f'Sentence: "{it["sentence"]}"\n'
                       f'In this sentence, does "{it["pronoun"]}" refer to {it["candidate"]}?')

ap=argparse.ArgumentParser(); ap.add_argument("--system", required=True)
ap.add_argument("--prefix", default=None); ap.add_argument("--tag", required=True)
ap.add_argument("--mem-limit-gb", type=float, default=None,
                help="cap MLX allocation so it raises instead of wedging the machine")
a=ap.parse_args()
if a.mem_limit_gb:
    import mlx.core as _mx
    _mx.set_memory_limit(int(a.mem_limit_gb*1e9))
    print(f"[mem] MLX capped at {a.mem_limit_gb:.1f} GB", flush=True)

if a.system=="jev":
    from jevbench.adapters.base import http_post_json
    for line in open(".env"):
        if "=" in line and not line.startswith("#"):
            k,v=line.split("=",1); os.environ.setdefault(k.strip(),v.strip())
    KEY=os.environ["OPENROUTER_API_KEY"]
    def ask(state):
        body={"model":"typesafe/jev-1.13","state":state,
              "questions":{"d":{"type":"noul","instructions":INSTR,"criteria":CRIT}}}
        for k in range(4):
            try:
                st,r,_=http_post_json("https://openrouter.ai/api/v1/systemone",body,
                    {"Authorization":f"Bearer {KEY}","Content-Type":"application/json"},30)
                if st==200 and "answers" in r: return r["answers"]["d"]["noul"]
            except Exception: pass
            time.sleep(1.5*(k+1))
        return None
else:
    from ryotide.jevbench_adapter import MlxJevLocalAdapter
    @dataclass
    class T:
        id:str; state:str; question:dict; labels:list; expected:str
        family:str="wino"; split:str="public"; group:str=None
        provenance:dict=field(default_factory=dict)
    ad=MlxJevLocalAdapter(endpoint=a.system, orders=1, repeat=1,
                          pin_prefix=a.prefix, pin_marker="{}"); ad.load()
    def ask(state):
        r=ad.run(T(id="x",state=state,
                   question={"type":"noul","instructions":INSTR,"criteria":CRIT},
                   labels=["yes","no"],expected="yes"))
        return r.probs["yes"] if r.ok else None

rows=[]
for it in ITEMS:
    p=ask(state_of(it))
    pred = 1 if (p is not None and p>=0.5) else 0
    rows.append({**it, "p": None if p is None else round(p,3), "pred": pred,
                 "correct": pred==it["label"]})
acc=sum(r["correct"] for r in rows)/len(rows)
# twin consistency: both probes of a schema-pair resolved the SAME way = pattern match
pairs={}
for r in rows: pairs.setdefault(r["sentence"],[]).append(r)
both=sum(1 for v in pairs.values() if all(x["correct"] for x in v))
schemas={}
for r in rows: schemas.setdefault(r["schema"],[]).append(r)
twin_ok=sum(1 for v in schemas.values() if all(x["correct"] for x in v))
print(f"\n=== {a.tag} ===  {len(rows)} probes over 5 twin schemas")
print(f"  item accuracy        {acc:.3f}  ({sum(r['correct'] for r in rows)}/{len(rows)})")
print(f"  sentences fully right {both}/{len(pairs)}")
print(f"  TWIN PAIRS fully right {twin_ok}/{len(schemas)}   <- both halves, the real test")
for r in rows:
    mark="ok " if r["correct"] else "MISS"
    print(f"   {mark} p={r['p']}  \"{r['pronoun']}\" -> {r['candidate']:24s} (gold {'yes' if r['label'] else 'no ' })"
          f"  {r['sentence'][:58]}...")
json.dump(rows, open(f"results/wino_{a.tag}.json","w"), indent=1)
