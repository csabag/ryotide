"""Synthetic intent routing with large option menus (20-151), labels computed by code.

Shaped like CLINC150 / BANKING77: a customer message, pick its intent from a menu.
150 intents (15 objects x 10 actions) plus out_of_scope. Messages use varied
phrasings, and some mention a second object in passing as a distractor.
Options are listed alphabetically, as they arrive over the TypeSafe wire.

This checks the MACHINERY of many-option reads (valid codes, sane probabilities,
accuracy well above chance). Keyword-to-intent matching is easier than real
benchmarks; it says nothing about how hard those are.

    uv run python bench/synthetic/gen_many_options.py --per-size 30 --seed 11
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "jevbench"))
from jevbench.tasks import Task  # noqa: E402

OBJECTS = {  # key: phrasings
    "card": ["my card", "my debit card", "the bank card"],
    "account": ["my account", "my current account", "the account"],
    "loan": ["my loan", "the personal loan", "my car loan"],
    "mortgage": ["my mortgage", "the home loan", "my mortgage agreement"],
    "transfer": ["a transfer", "the payment I sent", "my bank transfer"],
    "statement": ["my statement", "the monthly statement", "my e-statement"],
    "pin": ["my PIN", "the PIN code", "my card PIN"],
    "cheque": ["a cheque", "the cheque I deposited", "my cheque"],
    "direct_debit": ["a direct debit", "the direct debit for my gym", "my standing direct debit"],
    "savings_pot": ["my savings pot", "the savings space", "my rainy-day pot"],
    "overdraft": ["my overdraft", "the arranged overdraft", "my overdraft limit"],
    "travel_notice": ["a travel notice", "my travel notification", "the travel flag"],
    "address": ["my address", "my home address", "the postal address you have"],
    "email": ["my email", "the email address on file", "my contact email"],
    "phone": ["my phone number", "the mobile number", "my contact number"],
}
ACTIONS = {  # key: phrasings (object inserted at {o})
    "open": ["I'd like to open {o}", "please set up {o} for me", "can I start {o}"],
    "close": ["I want to close {o}", "please shut down {o}", "terminate {o} please"],
    "freeze": ["freeze {o} right now", "put a hold on {o}", "can you block {o} temporarily"],
    "replace": ["I need a replacement for {o}", "please send me a new one for {o}", "swap out {o}"],
    "dispute": ["I dispute {o}", "something is wrong with {o} and I want to challenge it", "raise a dispute about {o}"],
    "increase": ["increase {o}", "can you raise the limit on {o}", "I need more room on {o}"],
    "cancel": ["cancel {o}", "stop {o} from going through", "call off {o}"],
    "update": ["update {o}", "I need to change {o}", "please edit {o}"],
    "report_lost": ["I lost {o}", "{o} has gone missing", "I can't find {o} anywhere"],
    "check_status": ["what's the status of {o}", "any update on {o}", "where are we with {o}"],
}
OUT_OF_SCOPE = ["What's the weather like in Lisbon tomorrow?", "Can you recommend a good pizza place?",
                "How tall is the Eiffel Tower?", "Tell me a joke about cats.", "Who won the match last night?"]
ASIDES = ["By the way I also looked at {o} yesterday.", "Unrelated, but {o} is fine.", "I checked {o} too."]
INTENTS = [f"{a}_{o}" for o in OBJECTS for a in ACTIONS]          # 150


def describe(intent):
    if intent == "out_of_scope":
        return "The request is not about any banking task listed here."
    a, o = next((a, intent[len(a) + 1:]) for a in ACTIONS if intent.startswith(a + "_"))
    return f"The customer wants to {a.replace('_', ' ')} their {o.replace('_', ' ')}."


def message(rng, intent):
    if intent == "out_of_scope":
        return rng.choice(OUT_OF_SCOPE)
    a, o = next((a, intent[len(a) + 1:]) for a in ACTIONS if intent.startswith(a + "_"))
    text = rng.choice(ACTIONS[a]).format(o=rng.choice(OBJECTS[o])) + "."
    if rng.random() < 0.5:                                          # distractor object
        other = rng.choice([x for x in OBJECTS if x != o])
        aside = rng.choice(ASIDES).format(o=rng.choice(OBJECTS[other]))
        text = f"{aside} {text}" if rng.random() < 0.5 else f"{text} {aside}"
    return text[0].upper() + text[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-size", type=int, default=30)
    ap.add_argument("--sizes", default="20,26,27,40,77,151")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="data/synthetic/many-options-v1.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []
    for k in (int(x) for x in a.sizes.split(",")):
        for i in range(a.per_size):
            gold = "out_of_scope" if (k == 151 and rng.random() < 0.1) else rng.choice(INTENTS)
            pool = INTENTS + ["out_of_scope"]
            menu = set([gold]) | set(rng.sample([x for x in pool if x != gold], k - 1))
            labels = sorted(menu)                                   # alphabetical, as on the wire
            rows.append(dict(
                id=f"many-k{k:03d}-{i:03d}", family=f"many_k{k}", split="public", group=f"many-k{k}",
                state=f"Customer message: \"{message(rng, gold)}\"",
                labels=labels, expected=gold,
                question={"type": "choice", "instructions": "Which intent does the customer message express?",
                          "criteria": {l: describe(l) for l in labels}},
                provenance={"generator": "bench/synthetic/gen_many_options.py", "license": "MIT"}))
    for r in rows:
        Task.from_dict(r)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} tasks, sizes {a.sizes}, {a.per_size} each -> {a.out}")


if __name__ == "__main__":
    main()
