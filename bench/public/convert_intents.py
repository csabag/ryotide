"""Public intent benchmarks as JevBench tasks: CLINC150 (plus, test) and BANKING77 (test).

A sanity check of large menus on real data, not a tuning set: nothing here was
used to choose any setting. Each utterance becomes one choice question whose
options are the dataset's intent names, sorted the way the TypeSafe wire sorts
them; the criteria are just the names in words, since the datasets ship none.

    uv run python bench/public/convert_intents.py      # -> data/public/{clinc150,banking77}.jsonl

CLINC's out-of-scope class is named "oos". Under that literal name the models
never chose it (Gemma: 0/164 out-of-scope items), so the -oosdesc variant
describes it in words; a real request would describe such an option. That is a
post-hoc change to one label's description, reported next to the literal run.

It also writes a seeded random sample of each (-s1000.jsonl, seed 0): enough
to measure accuracy to about +-2.5 points, a fifth of the full run time.
"""
from __future__ import annotations

import csv
import io
import json
import os
import random
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "jevbench"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from jevbench.tasks import Task  # noqa: E402
from ryotide.server import natural_key  # noqa: E402

SAMPLE, SEED = 1000, 0
OOS_DESCRIPTION = "out of scope: the message matches none of the other intents"
BANKING77_TEST = ("https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
                  "master/banking_data/test.csv")


def task(name, i, text, gold, names, describe=None):
    labels = sorted(names, key=natural_key)
    describe = describe or {}
    return dict(
        id=f"{name}-{i:05d}", family=name, split="public", group=name,
        state=f"Customer message: \"{text}\"", labels=labels, expected=gold,
        question={"type": "choice", "instructions": "Which intent does the customer message express?",
                  "criteria": {l: describe.get(l, l.replace("_", " ")) for l in labels}},
        provenance={"source": name, "converter": "bench/public/convert_intents.py"})


def clinc150(describe=None):
    from datasets import load_dataset
    ds = load_dataset("clinc/clinc_oos", "plus", split="test")
    names = ds.features["intent"].names                      # 151 incl. "oos"
    return [task("clinc150", i, r["text"], names[r["intent"]], names, describe) for i, r in enumerate(ds)]


def banking77():
    rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(BANKING77_TEST).read().decode())))
    names = sorted({r["category"] for r in rows})           # 77
    return [task("banking77", i, r["text"], r["category"], names) for i, r in enumerate(rows)]


def main():
    os.makedirs("data/public", exist_ok=True)
    for name, fn in (("clinc150", clinc150), ("banking77", banking77),
                     ("clinc150-oosdesc", lambda: clinc150({"oos": OOS_DESCRIPTION}))):
        rows = fn()
        for r in rows:
            Task.from_dict(r)
        sample = sorted(random.Random(SEED).sample(rows, SAMPLE), key=lambda r: r["id"])
        for out, rs in ((name, rows), (f"{name}-s{SAMPLE}", sample)):
            with open(f"data/public/{out}.jsonl", "w") as fh:
                for r in rs:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{out}: {len(rs)} tasks, {len(rs[0]['labels'])} options -> data/public/{out}.jsonl")


if __name__ == "__main__":
    main()
