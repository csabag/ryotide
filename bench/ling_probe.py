"""Linguistic probe: the GLUE tasks that test language itself rather than
rubric application over facts.

  cola  grammatical acceptability   -> Matthews corr (accuracy misleads at 67% majority)
  mrpc  semantic equivalence        -> accuracy + F1
  wnli  Winograd coreference        -> accuracy (tiny, 71 items, notoriously adversarial)

Same items for every system, local and hosted.
"""
import argparse, json, math, os, random, sys, time
from dataclasses import dataclass, field
sys.path.insert(0, "src")

def mcc(y, p):
    tp=sum(1 for a,b in zip(y,p) if a==1 and b==1); tn=sum(1 for a,b in zip(y,p) if a==0 and b==0)
    fp=sum(1 for a,b in zip(y,p) if a==0 and b==1); fn=sum(1 for a,b in zip(y,p) if a==1 and b==0)
    d=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return 0.0 if d==0 else (tp*tn-fp*fn)/d
def f1(y,p):
    tp=sum(1 for a,b in zip(y,p) if a==1 and b==1); fp=sum(1 for a,b in zip(y,p) if a==0 and b==1)
    fn=sum(1 for a,b in zip(y,p) if a==1 and b==0)
    return 0.0 if tp==0 else 2*tp/(2*tp+fp+fn)

SPEC = {
 "cola": ("cola", lambda e: f'Sentence: "{e["sentence"].strip()}"',
   "Is this sentence grammatically acceptable English?",
   {"true":"Grammatically acceptable.","false":"Not grammatically acceptable."}, 1),
 "mrpc": ("mrpc", lambda e: f'Sentence 1: "{e["sentence1"].strip()}"\nSentence 2: "{e["sentence2"].strip()}"',
   "Do these two sentences mean the same thing?",
   {"true":"The sentences are semantically equivalent.","false":"The sentences differ in meaning."}, 1),
 "wnli": ("wnli", lambda e: f'Sentence: "{e["sentence1"].strip()}"\nStatement: "{e["sentence2"].strip()}"',
   "Given the sentence, is the statement true?",
   {"true":"The statement follows from the sentence.","false":"The statement does not follow."}, 1),
}
ap=argparse.ArgumentParser()
ap.add_argument("--system", required=True, help="jev | <mlx model path>")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--prefix", default=None); ap.add_argument("--tag", required=True)
a=ap.parse_args()
from datasets import load_dataset

if a.system == "jev":
    from jevbench.adapters.base import http_post_json
    for line in open(".env"):
        if "=" in line and not line.startswith("#"):
            k,v=line.split("=",1); os.environ.setdefault(k.strip(), v.strip())
    KEY=os.environ["OPENROUTER_API_KEY"]
    def ask(state, instr, crit):
        body={"model":"typesafe/jev-1.13","state":state,
              "questions":{"d":{"type":"noul","instructions":instr,"criteria":crit}}}
        for attempt in range(4):        # transient network faults killed a whole run
            try:
                st,r,_=http_post_json("https://openrouter.ai/api/v1/systemone",body,
                    {"Authorization":f"Bearer {KEY}","Content-Type":"application/json"},30)
                if st==200 and isinstance(r,dict) and "answers" in r:
                    return r["answers"]["d"]["noul"]
            except Exception:
                pass
            time.sleep(1.5*(attempt+1))
        return None
else:
    from ryotide.jevbench_adapter import MlxJevLocalAdapter
    @dataclass
    class T:
        id:str; state:str; question:dict; labels:list; expected:str
        family:str="glue"; split:str="public"; group:str=None
        provenance:dict=field(default_factory=dict)
    ad=MlxJevLocalAdapter(endpoint=a.system, orders=1, repeat=1,
                          pin_prefix=a.prefix, pin_marker="{}"); ad.load()
    def ask(state, instr, crit):
        t=T(id="x", state=state, question={"type":"noul","instructions":instr,"criteria":crit},
            labels=["yes","no"], expected="yes")
        r=ad.run(t)
        return r.probs["yes"] if r.ok else None

out={}
for name,(cfg,state_of,instr,crit,true_lab) in SPEC.items():
    ds=load_dataset("nyu-mll/glue",cfg,split="validation")
    idx=list(range(len(ds))); random.Random(0).shuffle(idx); idx=idx[:a.n]
    y,p=[],[]; t0=time.time()
    print(f"  [{name}] starting, {len(idx)} items", flush=True)
    for k,i in enumerate(idx,1):
        e=ds[i]; pr=ask(state_of(e),instr,crit)
        if pr is None: continue
        y.append(int(e["label"])); p.append(true_lab if pr>=0.5 else 1-true_lab)
        if k % 25 == 0:
            run=sum(1 for u,v in zip(y,p) if u==v)/len(y)
            el=time.time()-t0
            print(f"  [{name}] {k}/{len(idx)}  running acc {run:.3f}  "
                  f"{el:.0f}s  eta {el/k*(len(idx)-k):.0f}s", flush=True)
    acc=sum(1 for u,v in zip(y,p) if u==v)/max(len(y),1)
    maj=max(y.count(0),y.count(1))/max(len(y),1)
    out[name]={"n":len(y),"accuracy":round(acc,4),"majority":round(maj,4),
               "mcc":round(mcc(y,p),4),"f1":round(f1(y,p),4),"s":round(time.time()-t0,1)}
    print(f"  {name:5s} n={len(y):3d} acc {acc:.3f} (maj {maj:.3f})  MCC {out[name]['mcc']:+.3f}  "
          f"F1 {out[name]['f1']:.3f}  [{out[name]['s']}s]", flush=True)
json.dump(out, open(f"results/ling_{a.tag}.json","w"), indent=1)
