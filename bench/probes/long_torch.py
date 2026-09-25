"""Long inputs with a coded 40-option menu on CUDA (torch digits reader): peak memory, time, validity."""
import sys, time, random, types, torch
sys.path.insert(0, "src")
from ryotide.jevbench_adapter import MlxJevLocalAdapter
model, rev = sys.argv[1], sys.argv[2]
ad = MlxJevLocalAdapter(endpoint=model, revision=rev, backend="torch", device="cuda", orders=1, repeat=2,
                        pin_prefix="Answer: **", pin_marker="{}")
clf = ad.load(); tok = clf.tokenizer
print(f"{model}: weights loaded, {torch.cuda.memory_allocated()/1e9:.1f} GB on GPU; attention impl {clf.model.config._attn_implementation}", flush=True)
rng = random.Random(3)
filler = ("The quarterly operations review covered inventory, staffing, vendor contracts and "
          "facility maintenance across all regional sites. ")
labels = [f"tool_{i:02d}" for i in range(40)]
crit = {l: f"Retrieves {l.replace('_', ' ')} records from the {rng.choice(['billing','ledger','crm','hr','ops'])} system." for l in labels}
for target in (8000, 16000, 32000, 64000):
    body = filler * max(1, target // len(tok.encode(filler)))
    state = body[: len(body) // 2] + "The request is to retrieve tool 27 records from the system. " + body[len(body) // 2:]
    t = types.SimpleNamespace(state=state, labels=labels, task_id="long",
                              question={"type": "choice", "instructions": "Which tool fits the request described in the document?", "criteria": crit})
    toks = ad._tokens(ad._prompt(t, list(range(40))))
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        t0 = time.perf_counter(); p, mass = ad._read_codes(toks, 40); torch.cuda.synchronize(); dt = time.perf_counter() - t0
        top = labels[max(range(40), key=lambda i: p[i])]
        print(f"  state ~{target:>6} | prompt {len(toks):6d} tok | {dt:5.1f}s | peak GPU {torch.cuda.max_memory_allocated()/1e9:5.1f} GB | "
              f"mass {mass:.3f} | top {top} p={max(p):.2f}", flush=True)
    except torch.OutOfMemoryError:
        print(f"  state ~{target:>6} | prompt {len(toks):6d} tok | OUT OF GPU MEMORY (peak before failure {torch.cuda.max_memory_allocated()/1e9:.1f} GB)", flush=True)
        torch.cuda.empty_cache()
