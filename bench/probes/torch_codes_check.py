"""Torch digits reader (prompt cache kept, copied per group) vs a slow reference
(fresh full forward over prompt + letter for every group). Must agree up to bf16 noise."""
import sys, json, time
sys.path.insert(0, "src"); sys.path.insert(0, "vendor/jevbench")
from ryotide.jevbench_adapter import MlxJevLocalAdapter
from jevbench.tasks import load_jsonl
model, rev = sys.argv[1], sys.argv[2]
ad = MlxJevLocalAdapter(endpoint=model, revision=rev, backend="torch", device="mps", orders=1, repeat=2,
                        pin_prefix="Answer: **", pin_marker="{}")
clf = ad.load()
tasks = [t for t in load_jsonl("data/synthetic/many-options-mid.jsonl") if len(t.labels) in (27, 40)][:4]
worst_group = worst_digit = 0.0; same_top = 0
for t in tasks:
    n = len(t.labels); toks = ad._tokens(ad._prompt(t, list(range(n))))
    letter_ids, digit_ids = ad._code_ids(n)
    t0 = time.perf_counter(); pg, fg, pd, md = clf.read_codes(toks, letter_ids, digit_ids); fast = time.perf_counter() - t0
    t0 = time.perf_counter()
    ref_g, _ = clf.read(toks, letter_ids)
    ref_d = [clf.read(list(toks) + [lid], dids)[0] for lid, dids in zip(letter_ids, digit_ids)]
    slow = time.perf_counter() - t0
    worst_group = max(worst_group, max(abs(a - b) for a, b in zip(pg, ref_g)))
    worst_digit = max(worst_digit, max(abs(a - b) for x, y in zip(pd, ref_d) for a, b in zip(x, y)))
    joint = lambda G, Dg: [g * d for g, ds in zip(G, Dg) for d in ds]
    jf, jr = joint(pg, pd), joint(ref_g, ref_d)
    same_top += jf.index(max(jf)) == jr.index(max(jr))
    print(f"  {t.id} ({n} opts, {len(letter_ids)} groups): fast {fast:.1f}s vs reference {slow:.1f}s", flush=True)
print(f"{model}: max |dP| letters {worst_group:.2e}, digits {worst_digit:.2e}; same top option {same_top}/{len(tasks)}")
