"""Long inputs with a coded 40-option menu: latency, peak memory, read validity (MLX)."""
import sys, time, random, types
sys.path.insert(0, "src")
import mlx.core as mx
from ryotide.jevbench_adapter import MlxJevLocalAdapter
ad = MlxJevLocalAdapter(endpoint=sys.argv[1], orders=1, repeat=2, pin_prefix="Answer: **", pin_marker="{}")
clf = ad.load(); tok = clf.tokenizer
rng = random.Random(3)
filler = ("The quarterly operations review covered inventory, staffing, vendor contracts and "
          "facility maintenance across all regional sites. ")
labels = [f"tool_{i:02d}" for i in range(40)]
crit = {l: f"Retrieves {l.replace('_', ' ')} records from the {rng.choice(['billing','ledger','crm','hr','ops'])} system." for l in labels}
for target in (8000, 16000, 32000):
    reps = max(1, target // len(tok.encode(filler)))
    body = filler * reps
    needle = "The request is to retrieve tool 27 records from the system."
    state = body[: len(body) // 2] + needle + " " + body[len(body) // 2:]
    t = types.SimpleNamespace(state=state, labels=labels, task_id="long",
                              question={"type": "choice", "instructions": "Which tool fits the request described in the document?",
                                        "criteria": crit})
    toks = ad._tokens(ad._prompt(t, list(range(40))))
    mx.reset_peak_memory()
    t0 = time.perf_counter(); p, mass = ad._read_codes(toks, 40); dt = time.perf_counter() - t0
    top = labels[max(range(40), key=lambda i: p[i])]
    print(f"state ~{target:>6} tok | prompt {len(toks):6d} tok | {dt:6.1f}s | peak GPU mem {mx.get_peak_memory()/1e9:5.1f} GB | "
          f"marker mass {mass:.3f} | top {top} (needle: tool_27) p={max(p):.2f}", flush=True)
