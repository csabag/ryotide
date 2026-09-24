"""Synthetic typed decisions for testing WHEN question repetition (the echo) helps.

Every label is computed by code from the generated state -- correct by
construction, never judged by a model. Each scenario is posed in several forms so
content is held fixed while the question form varies:

  noul      yes/no: "is the answer <candidate>?"  (candidate = gold half the time)
  choice2   two named alternatives (gold + one distractor)
  choice4   four named alternatives, authored order
  choice4r1, choice4r2   the same four, cyclically rotated (order-invariance test)

Three families vary what the decision needs:
  lateness     arithmetic on timestamps into ordinal buckets
  routing      departments with overlapping, precedence-ordered scopes
  eligibility  a rule with an exception and an exception to the exception

Output is JevBench task JSONL (validated with the harness's own Task schema).

    uv run python bench/synthetic/gen_echo_tasks.py --n 40 --seed 7 --out data/synthetic/echo-v1.jsonl
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "jevbench"))
from jevbench.tasks import Task  # noqa: E402

# --------------------------------------------------------------------- lateness
LATE = [
    ("early_or_on_time", "Delivered at or before the end of the promised window."),
    ("late_under_30m", "Delivered after the window, by less than 30 minutes."),
    ("late_30m_to_2h", "Delivered 30 minutes to under 2 hours after the window."),
    ("late_2h_to_6h", "Delivered 2 hours to under 6 hours after the window."),
    ("late_6h_or_more", "Delivered 6 hours or more after the window."),
]
LATE_BOUNDS = [(-600, 0), (1, 29), (31, 119), (121, 359), (361, 1500)]  # never on a boundary


def gen_lateness(rng: random.Random, i: int) -> tuple[str, str, str]:
    base = dt.datetime(2026, 3, rng.randint(2, 26), 0, 0)
    ids = rng.sample(range(10000, 99999), rng.randint(7, 11))
    target = rng.choice(ids)
    gold_idx = rng.randrange(len(LATE))
    lines = ["CARRIER DELIVERY LOG (times local; windows as promised to the customer)", ""]
    for sid in ids:
        start = base + dt.timedelta(minutes=rng.randrange(7 * 60, 18 * 60, 15))
        end = start + dt.timedelta(minutes=rng.choice([60, 120, 180, 240]))
        lo, hi = LATE_BOUNDS[gold_idx] if sid == target else LATE_BOUNDS[rng.randrange(len(LATE))]
        actual = end + dt.timedelta(minutes=rng.randint(lo, hi))
        note = rng.choice(["", " (left with neighbour)", " (signature on file)", " (re-attempt)", ""])
        lines.append(f"Shipment S-{sid}: window {start:%a %d %b %H:%M}-{end:%H:%M}; "
                     f"delivered {actual:%a %d %b %H:%M}{note}")
    q = f"Classify how late shipment S-{target} was delivered relative to its promised window."
    return "\n".join(lines), q, LATE[gold_idx][0]


# ---------------------------------------------------------------------- routing
ROUTE = [
    ("fraud", "The customer says they never authorised the charge."),
    ("disputes", "The customer has already reported the charge to their bank."),
    ("refunds", "Money back for a charge made within the last 30 days."),
    ("billing", "Invoices, payment methods, and charges older than 30 days."),
    ("account", "Login, password or profile changes that involve no money."),
]
ROUTE_PRECEDENCE = ("Precedence when more than one scope applies: fraud, then disputes, then "
                    "refunds or billing by the age of the charge. Account only when no charge is involved.")


def gen_routing(rng: random.Random, i: int) -> tuple[str, str, str]:
    today = dt.date(2026, 3, rng.randint(1, 28))
    kind = rng.choice(["charge", "charge", "charge", "account"])
    unauth = kind == "charge" and rng.random() < 0.3
    bank = kind == "charge" and rng.random() < 0.3
    age = rng.choice([rng.randint(2, 28), rng.randint(32, 90)])
    when = today - dt.timedelta(days=age)
    amount = rng.choice([19.99, 42.50, 129.00, 7.25, 310.40])
    if kind == "account":
        gold = "account"
        msg = rng.choice(["I can't log in since the password reset email never arrives.",
                          "Please change the email address on my profile to my work one.",
                          "My account shows the wrong name, can you fix it?"])
    else:
        gold = "fraud" if unauth else "disputes" if bank else ("refunds" if age <= 30 else "billing")
        bits = [f"There is a charge of ${amount:.2f} on {when:%d %B %Y}."]
        bits.append("I don't recognise it and never agreed to it." if unauth
                    else "It was for the annual plan I cancelled, I want it back.")
        if bank:
            bits.append("I already opened a case with my bank about it.")
        rng.shuffle(bits)
        msg = " ".join(bits)
    lines = [f"Today is {today:%d %B %Y}.", "", "SUPPORT DIRECTORY"]
    lines += [f"- {k}: {v}" for k, v in ROUTE] + [ROUTE_PRECEDENCE, "", "RECENT TICKETS"]
    others = rng.randint(3, 6)
    tickets = [("T-%d" % rng.randint(1000, 9999),
                rng.choice(["Where is my parcel?", "Can I get an invoice copy for February?",
                            "The app crashes on start.", "I want to upgrade my plan."]))
               for _ in range(others)]
    target = "T-%d" % rng.randint(1000, 9999)
    tickets.insert(rng.randrange(len(tickets) + 1), (target, msg))
    lines += [f"{t}: {m}" for t, m in tickets]
    q = f"Which team should handle ticket {target}?"
    return "\n".join(lines), q, gold


# ------------------------------------------------------------------ eligibility
ELIG = [
    ("approve", "Meets the standard rule or its long-tenure exception, amount within limits."),
    ("approve_capped", "Eligible, but the amount must be capped at 3,000."),
    ("manual_review", "Must go to a human reviewer."),
    ("deny", "Not eligible under the policy."),
]
POLICY = ("CREDIT POLICY\n"
          "1. Standard: eligible if tenure is at least 12 months and there were no late payments "
          "in the last 6 months.\n"
          "2. Exception: members with at least 36 months of tenure remain eligible with up to 2 "
          "late payments in the last 6 months.\n"
          "3. Amounts above 5,000 always go to manual review, unless tenure is at least 60 months.\n"
          "4. Eligible members with less than 24 months of tenure are capped at 3,000.\n"
          "5. Anyone not eligible under 1 or 2 is denied (rule 3 still applies first).")


def elig_outcome(tenure: int, late: int, amount: int) -> str:
    if amount > 5000 and tenure < 60:
        return "manual_review"
    eligible = (tenure >= 12 and late == 0) or (tenure >= 36 and late <= 2)
    if not eligible:
        return "deny"
    if tenure < 24 and amount > 3000:
        return "approve_capped"
    return "approve"


def gen_eligibility(rng: random.Random, i: int) -> tuple[str, str, str]:
    want = rng.choice([k for k, _ in ELIG])
    for _ in range(10000):                                  # sample until the gold is `want`
        t, l, a = rng.randint(3, 80), rng.choice([0, 0, 1, 2, 3]), rng.choice(
            [800, 1500, 2500, 3500, 4200, 4800, 5600, 7200])
        if elig_outcome(t, l, a) == want:
            break
    names = rng.sample(["Avery", "Blake", "Casey", "Devon", "Emery", "Finley", "Harper",
                        "Jordan", "Kendall", "Logan", "Morgan", "Quinn", "Reese", "Sawyer"], 6)
    target = names[rng.randrange(len(names))]
    rows = []
    for n in names:
        tt, ll, aa = (t, l, a) if n == target else (rng.randint(3, 80), rng.choice([0, 0, 1, 2, 3]),
                                                     rng.choice([800, 2500, 3500, 5600, 7200]))
        rows.append(f"{n}: tenure {tt} months; late payments (last 6 months): {ll}; requested {aa:,}")
    state = POLICY + "\n\nAPPLICATIONS\n" + "\n".join(rows)
    return state, f"Decide the outcome of {target}'s application.", elig_outcome(t, l, a)


FAMILIES = {"lateness": (gen_lateness, LATE), "routing": (gen_routing, ROUTE),
            "eligibility": (gen_eligibility, ELIG)}


def tasks_for(family: str, i: int, rng: random.Random) -> list[dict]:
    gen, options = FAMILIES[family]
    state, q, gold = gen(rng, i)
    crit = dict(options)
    order = [k for k, _ in options]                         # authored order
    others = [k for k in order if k != gold]
    pick2 = sorted([gold, rng.choice(others)], key=order.index)
    pick4 = sorted([gold] + rng.sample(others, 3), key=order.index)
    cand = gold if rng.random() < 0.5 else rng.choice(others)
    base = dict(family=f"syn_{family}", state=state, split="public", group=f"syn-{family}-{i:03d}",
                provenance={"generator": "bench/synthetic/gen_echo_tasks.py", "license": "MIT"})
    out = [dict(base, id=f"syn-{family}-{i:03d}-noul", labels=["no", "yes"],
                expected="yes" if cand == gold else "no",
                question={"type": "noul",
                          "instructions": f"{q} Is the correct category '{cand}' ({crit[cand]})?",
                          "criteria": {"true": f"The correct category is {cand}.",
                                       "false": f"The correct category is not {cand}."}})]
    for form, labels in (("choice2", pick2), ("choice4", pick4),
                         ("choice4r1", pick4[1:] + pick4[:1]), ("choice4r2", pick4[2:] + pick4[:2])):
        out.append(dict(base, id=f"syn-{family}-{i:03d}-{form}", labels=labels, expected=gold,
                        question={"type": "choice", "instructions": q,
                                  "criteria": {k: crit[k] for k in labels}}))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="scenarios per family")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/synthetic/echo-v1.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = [t for fam in FAMILIES for i in range(a.n) for t in tasks_for(fam, i, rng)]
    for r in rows:
        Task.from_dict(r)                                    # harness schema validation
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} tasks ({a.n} scenarios x {len(FAMILIES)} families x 5 forms) -> {a.out}")


if __name__ == "__main__":
    main()
