"""Synthetic yes/no and score decisions for JevBench v1.6's decisiveness rule.

v1.6 counts a yes/no answer only when P(yes) >= 0.8 or <= 0.2, and scores ordinal
("score") questions on their own. This set holds both types, in the scenarios of
gen_echo_tasks.py (lateness, routing, eligibility), with every label computed by code
from the generated state -- never judged by a model, never taken from benchmark items.

  noul_cat   "Is the correct category <candidate>?"        (as in echo-v1)
  noul_fact  a direct yes/no about the target, e.g. "Was S-123 delivered late?"
  score_sev  lateness severity, levels 0-4 (the five lateness buckets, in order)
  score_cnt  how many shipments in the log were 6 hours or more late: 0, 1, 2, 3 or more

Yes and no are balanced per family. Fresh seed, so no overlap with echo-v1.

    uv run python bench/synthetic/gen_decisive_tasks.py --n 60 --seed 4242 --out data/synthetic/decide-v1.jsonl
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "jevbench"))
from gen_echo_tasks import (ELIG, LATE, ROUTE, gen_eligibility, gen_lateness,  # noqa: E402
                            gen_routing)
from jevbench.tasks import Task  # noqa: E402

FACT = {   # family -> (gold category -> (yes/no question about the target, answer is yes?))
    "lateness": lambda g, tgt: rng_pick([
        (f"Was shipment {tgt} delivered after the end of its promised window?", g != "early_or_on_time"),
        (f"Was shipment {tgt} delivered 2 hours or more after the end of its window?",
         g in ("late_2h_to_6h", "late_6h_or_more")),
        (f"Was shipment {tgt} delivered less than 30 minutes after the end of its window "
         "(or on time)?", g in ("early_or_on_time", "late_under_30m"))]),
    "routing": lambda g, tgt: rng_pick([
        (f"Does ticket {tgt} involve money (a charge)?", g != "account"),
        (f"Should ticket {tgt} be handled by the fraud team?", g == "fraud"),
        (f"Should ticket {tgt} go to refunds or billing?", g in ("refunds", "billing"))]),
    "eligibility": lambda g, tgt: rng_pick([
        (f"Is {tgt}'s application approved (capped or not)?", g in ("approve", "approve_capped")),
        (f"Must {tgt}'s application go to a human reviewer?", g == "manual_review"),
        (f"Is {tgt}'s application denied?", g == "deny")]),
}
_RNG: random.Random


def rng_pick(options):
    return _RNG.choice(options)


def target_of(family: str, q: str) -> str:
    if family == "eligibility":
        return re.search(r"of (\w+)'s application", q).group(1)
    return re.search(r"(S-\d+|T-\d+)", q).group(1)


def late_count(state: str) -> int:
    """Shipments delivered >= 6 h after their window, recomputed from the log text."""
    n = 0
    for m in re.finditer(r"window (\w{3} \d\d \w{3}) (\d\d:\d\d)-(\d\d:\d\d); delivered (\w{3} \d\d \w{3}) (\d\d:\d\d)",
                         state):
        day0, _, end, day1, at = m.groups()
        e = dt.datetime.strptime(f"2026 {day0} {end}", "%Y %a %d %b %H:%M")
        d = dt.datetime.strptime(f"2026 {day1} {at}", "%Y %a %d %b %H:%M")
        n += (d - e) >= dt.timedelta(hours=6)
    return n


def rows_for(family: str, i: int, rng: random.Random, want_yes: bool) -> list[dict]:
    gen, options = {"lateness": (gen_lateness, LATE), "routing": (gen_routing, ROUTE),
                    "eligibility": (gen_eligibility, ELIG)}[family]
    for _ in range(200):                     # resample the fact question until its answer = want_yes
        state, q, gold = gen(rng, i)
        tgt = target_of(family, q)
        fq, fyes = FACT[family](gold, tgt)
        if fyes == want_yes:
            break
    crit = dict(options)
    others = [k for k, _ in options if k != gold]
    cand = gold if want_yes else rng.choice(others)
    base = dict(family=f"dec_{family}", state=state, split="public", group=f"dec-{family}-{i:03d}",
                provenance={"generator": "bench/synthetic/gen_decisive_tasks.py", "license": "MIT"})
    out = [dict(base, id=f"dec-{family}-{i:03d}-noul_cat", labels=["no", "yes"],
                expected="yes" if cand == gold else "no",
                question={"type": "noul", "instructions": f"{q} Is the correct category '{cand}' ({crit[cand]})?",
                          "criteria": {"true": f"The correct category is {cand}.",
                                       "false": f"The correct category is not {cand}."}}),
           dict(base, id=f"dec-{family}-{i:03d}-noul_fact", labels=["no", "yes"],
                expected="yes" if fyes else "no",
                question={"type": "noul", "instructions": fq})]
    if family == "lateness":
        out.append(dict(base, id=f"dec-{family}-{i:03d}-score_sev", labels=["0", "1", "2", "3", "4"],
                        expected=[k for k, _ in LATE].index(gold),
                        question={"type": "score", "instructions": f"Rate how late shipment {tgt} was "
                                  "delivered relative to its promised window.",
                                  "criteria": [d for _, d in LATE]}))
        c = late_count(state)
        out.append(dict(base, id=f"dec-{family}-{i:03d}-score_cnt", labels=["0", "1", "2", "3"],
                        expected=min(c, 3),
                        question={"type": "score", "instructions": "How many shipments in the log were "
                                  "delivered 6 hours or more after the end of their promised window?",
                                  "criteria": ["None", "Exactly one", "Exactly two", "Three or more"]}))
    return out


def main() -> None:
    global _RNG
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="scenarios per family")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default="data/synthetic/decide-v1.jsonl")
    a = ap.parse_args()
    rng = _RNG = random.Random(a.seed)
    rows = [t for fam in FACT for i in range(a.n) for t in rows_for(fam, i, rng, want_yes=i % 2 == 0)]
    for r in rows:
        Task.from_dict(r)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds = {}
    for r in rows:
        k = r["id"].rsplit("-", 1)[1]
        kinds[k] = kinds.get(k, 0) + 1
    print(f"wrote {len(rows)} tasks -> {a.out}: {kinds}")


if __name__ == "__main__":
    main()
