#!/usr/bin/env python3
"""Offline JevBench cascade/committee/best-of-n recombination.

The input manifest maps each published system key to its frozen JevBench JSONL
files.  Task text is never written to the output; only aggregate curves and
content hashes are emitted.
"""
from __future__ import annotations

import argparse, hashlib, itertools, json, math
from collections import defaultdict
from pathlib import Path

TIER_WEIGHTS = {"easy": .14, "standard": .28, "judge": .28, "hard": .30}


def percentile(xs, q):
    xs = sorted(xs)
    k = (len(xs)-1)*q; a = math.floor(k); b = math.ceil(k)
    return xs[a] if a == b else xs[a]*(b-k)+xs[b]*(k-a)


def ece(rows, bins=10):
    groups = [[] for _ in range(bins)]
    for r in rows:
        conf = max(r["probs"].values()); pred = max(r["probs"], key=r["probs"].get)
        groups[min(int(conf*bins), bins-1)].append((conf, pred == r["expected"]))
    n = len(rows)
    return sum(len(g)/n * abs(sum(c for c,_ in g)/len(g)-sum(ok for _,ok in g)/len(g)) for g in groups if g)


def scores(rows, cost, endpoint_adjusted=True):
    by = defaultdict(list)
    for r in rows: by[r["tier"]].append(r)
    tiers = {t: sum(x["predicted"] == x["expected"] for x in rs)/len(rs) for t,rs in by.items()}
    intelligence = 100*sum(TIER_WEIGHTS[t]*v for t,v in tiers.items())/sum(TIER_WEIGHTS[t] for t in tiers)
    hard = by["hard"]
    hard_ece = ece(hard)
    prob = [r for r in hard if r.get("gold_probs")]
    tvd = sum(.5*sum(abs(r["probs"].get(k,0)-v) for k,v in r["gold_probs"].items()) for r in prob)/len(prob)
    calibration = (100*max(0,1-hard_ece/.5)+100*(1-tvd))/2
    p50, p95 = percentile([r["latency_s"] for r in rows],.5), percentile([r["latency_s"] for r in rows],.95)
    sp = lambda s: max(0,min(100,100-20*math.log10(s/.1)))
    speed = (sp(p50)+sp(p95))/2
    cost_axis=max(0,min(100,100-30*math.log10(cost/.001)))
    axes={"intelligence":intelligence,"calibration":calibration,"speed":speed,"cost":cost_axis}
    composite=math.exp(sum(.25*math.log(max(v,1)) for v in axes.values()))
    return {"tiers":tiers,"accuracy":sum(r["predicted"]==r["expected"] for r in rows)/len(rows),
            "ece_hard":hard_ece,"calibration":calibration,"p50_s":p50,"p95_s":p95,
            "cost_usd_per_1000":cost,"axes":axes,"jevbench_score":composite}


def load_jsonl(path):
    raw=Path(path).read_bytes()
    return [json.loads(x) for x in raw.splitlines() if x.strip()], hashlib.sha256(raw).hexdigest()


def combine(task, members, method, weights):
    labels=task["labels"]
    if method == "majority":
        votes=defaultdict(int)
        for r in members: votes[max(r["probs"],key=r["probs"].get)]+=1
        top=max(votes.values()); tied=[k for k,v in votes.items() if v==top]
        avg={k:sum(r["probs"].get(k,0) for r in members)/len(members) for k in labels}
        pred=max(tied,key=lambda k:(avg[k],-labels.index(k)))
        probs={k:v/len(members) for k,v in votes.items()}
        probs={k:probs.get(k,0) for k in labels}; probs[pred]+=max(0,1-sum(probs.values()))
    else:
        ws=[1]*len(members) if method=="probability_average" else weights
        probs={k:sum(w*r["probs"].get(k,0) for w,r in zip(ws,members))/sum(ws) for k in labels}
        pred=max(labels,key=lambda k:probs[k])
    return probs,pred


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("manifest"); ap.add_argument("output"); a=ap.parse_args()
    m=json.load(open(a.manifest)); tasks={}
    for spec in m["tasks"]:
        for x in load_jsonl(spec["path"])[0]:
            tasks[x["id"]]={**x,"tier":spec["tier"],"public":spec["public"]}
    systems={}; hashes={}
    for key,s in m["systems"].items():
        rs=[]; hashes[key]=[]
        for p in s["files"]:
            part,h=load_jsonl(p); rs+=part; hashes[key].append({"sha256":h,"rows":len(part)})
        d={}
        for r in rs:
            if r.get("ok") and r.get("probs"):
                t=tasks[r["task_id"]]; d[r["task_id"]]={**r,"tier":t["tier"],
                    "expected":t["expected"],"labels":t["labels"],"gold_probs":t.get("provenance",{}).get("gold_probs")}
        if len(d)==534: systems[key]=d
    published=m["published"]
    # Committees: all plausible 3/5 sets among systems with distributions, bounded to configured pool.
    committees=[]
    pool=[k for k in m["committee_pool"] if k in systems]
    for n in (3,5):
      for combo in itertools.combinations(pool,n):
       for method in ("majority","probability_average","calibration_weighted"):
        rows=[]
        for tid,t in tasks.items():
            mem=[systems[k][tid] for k in combo]
            probs,pred=combine(t,mem,method,[published[k]["calibration"]/100 for k in combo])
            rows.append({"tier":t["tier"],"expected":t["expected"],
              "labels":t["labels"],"gold_probs":t.get("provenance",{}).get("gold_probs"),"probs":probs,"predicted":pred,
              "latency_s":max(x["latency_s"]*published[k]["latency_factor"]+published[k]["latency_add_s"] for k,x in zip(combo,mem))})
        result=scores(rows,sum(published[k]["cost"] for k in combo)); committees.append({"members":combo,"method":method,**result})
    committees.sort(key=lambda x:x["jevbench_score"],reverse=True)
    # Cascades: threshold selection on public items, evaluation recorded for public and held-out.
    cascades=[]; thresholds=[i/100 for i in range(101)]
    for cheap,strong in itertools.permutations(m["cascade_pool"],2):
      if cheap not in systems or strong not in systems or published[cheap]["cost"]>=published[strong]["cost"]: continue
      cheap_pub=sum(r["predicted"]==r["expected"] for tid,r in systems[cheap].items() if tasks[tid]["public"])/231
      strong_pub=sum(r["predicted"]==r["expected"] for tid,r in systems[strong].items() if tasks[tid]["public"])/231
      # A cascade backend must actually be stronger on the tuning split; otherwise
      # threshold zero merely relabels the stronger cheap model as a "cascade".
      if strong_pub <= cheap_pub: continue
      curves=[]
      for th in thresholds:
       subsets={}
       for split,want_public in (("public",True),("heldout",False)):
        rows=[]; escalated=0
        for tid,t in tasks.items():
         if bool(t["public"])!=want_public: continue
         c,b=systems[cheap][tid],systems[strong][tid]; esc=max(c["probs"].values())<th; escalated+=esc; r=b if esc else c
         rows.append({"tier":t["tier"],"expected":t["expected"],"labels":t["labels"],
          "gold_probs":t.get("provenance",{}).get("gold_probs"),"probs":r["probs"],"predicted":r["predicted"],
          "latency_s":c["latency_s"]*published[cheap]["latency_factor"]+published[cheap]["latency_add_s"]+(b["latency_s"]*published[strong]["latency_factor"]+published[strong]["latency_add_s"] if esc else 0)})
        cost=published[cheap]["cost"]+escalated/len(rows)*published[strong]["cost"]
        subsets[split]={"n":len(rows),"escalation":escalated/len(rows),**scores(rows,cost)}
       curves.append({"threshold":th,**subsets})
      backend_acc=sum(r["predicted"]==r["expected"] for tid,r in systems[strong].items() if not tasks[tid]["public"])/303
      eligible=[x for x in curves if x["public"]["accuracy"]>=.99*strong_pub]
      chosen=min(eligible,key=lambda x:x["public"]["escalation"]) if eligible else curves[-1]
      h=chosen["heldout"]; useful=h["accuracy"]>=.99*backend_acc and h["cost_usd_per_1000"]<=.8*published[strong]["cost"] and h["p95_s"]<=published[strong]["p95"]
      th=chosen["threshold"]; full=[]; full_esc=0
      for tid,t in tasks.items():
        c,b=systems[cheap][tid],systems[strong][tid]; esc=max(c["probs"].values())<th; full_esc+=esc; r=b if esc else c
        full.append({"tier":t["tier"],"expected":t["expected"],"labels":t["labels"],"gold_probs":t.get("provenance",{}).get("gold_probs"),
          "probs":r["probs"],"predicted":r["predicted"],"latency_s":c["latency_s"]*published[cheap]["latency_factor"]+published[cheap]["latency_add_s"]+(b["latency_s"]*published[strong]["latency_factor"]+published[strong]["latency_add_s"] if esc else 0)})
      full_stats=scores(full,published[cheap]["cost"]+full_esc/len(full)*published[strong]["cost"])
      cascades.append({"cheap":cheap,"backend":strong,"chosen":chosen,"full":full_stats,"full_escalation":full_esc/len(full),"backend_heldout_accuracy":backend_acc,"useful":useful,"curve":curves})
    cascades.sort(key=lambda x:(x["useful"],-x["chosen"]["heldout"]["cost_usd_per_1000"],x["chosen"]["heldout"]["accuracy"]),reverse=True)
    bestn=[]
    for key,spec in m.get("best_of_n",{}).items():
        samples=[]; sample_hashes=[]
        for files in spec["samples"]:
            d={}; hs=[]
            for p in files:
                part,h=load_jsonl(p); hs.append(h)
                for r in part:
                    if r.get("ok") and r.get("probs"): d[r["task_id"]]=r
            samples.append(d); sample_hashes.append(hs)
        changed_probs=sum(len({json.dumps(s[tid]["probs"],sort_keys=True) for s in samples})>1 for tid in tasks)
        changed_labels=sum(len({s[tid]["predicted"] for s in samples})>1 for tid in tasks)
        rows=[]
        for tid,t in tasks.items():
            mem=[s[tid] for s in samples]; probs={lab:sum(r["probs"].get(lab,0) for r in mem)/len(mem) for lab in t["labels"]}
            rows.append({"tier":t["tier"],"expected":t["expected"],"labels":t["labels"],"gold_probs":t.get("provenance",{}).get("gold_probs"),
              "probs":probs,"predicted":max(t["labels"],key=lambda x:probs[x]),
              "latency_s":max(r["latency_s"]*published[key]["latency_factor"]+published[key]["latency_add_s"] for r in mem)})
        bestn.append({"system":key,"n":len(samples),"changed_probability_items":changed_probs,"changed_label_items":changed_labels,
          "sample_hashes":sample_hashes,**scores(rows,len(samples)*published[key]["cost"])})
    out={"schema":"jevbench-combinations/v1","benchmark":"JevBench v1.2.6","bar":m["bar"],"source_hashes":hashes,
         "complete_systems":sorted(systems),"committees":committees,"cascades":cascades,"best_of_n":bestn}
    Path(a.output).write_text(json.dumps(out,indent=2)+"\n")

if __name__ == "__main__": main()
