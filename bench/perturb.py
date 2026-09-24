"""Prepend one random token to the QUESTION BODY and watch what confidence does.

The state is left untouched -- only the instruction line that opens the question
gets one irrelevant word in front of it. The state is thousands of tokens, so a
junk token there would be diluted; the question is the short, load-bearing part
sitting right next to the read position, which is where a perturbation should
bite hardest if confidence is a surface artifact rather than a read on content.
"""
import json, random, sys
sys.path.insert(0, "src")
import mlx.core as mx
from ryotide.jevbench_adapter import MlxJevLocalAdapter
from jevbench.tasks import load_jsonl

N_TASKS, N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 40, 3
tasks = []
for f in ("original", "easy", "hard"):
    tasks += load_jsonl(f"vendor/jevbench/datasets/public/{f}.jsonl")
rng = random.Random(0)
sample = rng.sample(tasks, N_TASKS)

for MODEL, lbl in [("mlx-community/gemma-4-e4b-it-8bit", "Gemma 4 E4B"),
                   ("mlx-community/Qwen3.5-4B-MLX-8bit", "Qwen3.5-4B")]:
    ad = MlxJevLocalAdapter(endpoint=MODEL, orders=1); clf = ad.load()
    ad.calibrate(rng.sample(tasks, 6))
    tok = clf.tokenizer
    # plausible junk tokens: real words, no semantic relation to any task
    junk = ["banana", "Tuesday", "glass", "orbit", "velvet", "hammer"]
    deltas, flips, base_ok, pert_ok, n = [], 0, 0, 0, 0
    records = []
    for t in sample:
        k = len(t.labels)
        sel = mx.array(ad._letters(k))
        def read(prefix_word=None):
            # perturb the question's instruction line, never the state
            q = t.question
            saved = q["instructions"]
            if prefix_word is not None:
                q["instructions"] = prefix_word + " " + saved
            toks = ad._tokens(ad._prompt(t, list(range(k))))
            q["instructions"] = saved
            row = clf.logits_at_end(clf.prefill([]), toks)
            p = mx.softmax(row[sel].astype(mx.float32)).tolist()
            i = max(range(k), key=lambda j: p[j])
            return p[i], t.labels[i]
        c0, a0 = read()
        ok0 = (a0 == str(t.expected))
        base_ok += ok0; n += 1
        ds, fl = [], 0
        for s in range(N_SEEDS):
            c1, a1 = read(junk[(hash(t.id) + s) % len(junk)])
            deltas.append(c1 - c0); flips += (a1 != a0); pert_ok += (a1 == str(t.expected))
            ds.append(c1 - c0); fl += (a1 != a0)
        records.append({"id": t.id, "tier": ("easy" if t.id.startswith("easy-")
                        else "standard" if t.id.startswith("original-") else "hard"),
                        "conf": c0, "correct": bool(ok0), "flips": fl,
                        "mean_abs_d": sum(abs(d) for d in ds)/len(ds),
                        "max_abs_d": max(abs(d) for d in ds),
                        "mean_d": sum(ds)/len(ds)})
    m = len(deltas)
    absd = sorted(abs(d) for d in deltas)
    print(f"\n=== {lbl} === ({n} tasks x {N_SEEDS} junk tokens)")
    print(f"  baseline accuracy       {base_ok/n:.3f}")
    print(f"  perturbed accuracy      {pert_ok/m:.3f}")
    print(f"  prediction flipped on   {flips}/{m} = {flips/m:.1%}")
    print(f"  mean signed dconf       {sum(deltas)/m:+.4f}   (drift in confidence)")
    print(f"  median |dconf|          {absd[m//2]:.4f}")
    print(f"  90th pct |dconf|        {absd[int(m*0.9)]:.4f}   max {absd[-1]:.4f}")
    json.dump(records, open(f"results/jevbench/perturb_{lbl.replace(' ','_')}.json","w"), indent=1)
    del ad, clf
