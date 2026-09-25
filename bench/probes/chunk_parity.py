"""Chunked prefill vs the old single forward pass: same decisions, rounding-level differences."""
import sys, types, random, torch
sys.path.insert(0, "src"); sys.path.insert(0, "vendor/jevbench")
from ryotide.jevbench_adapter import MlxJevLocalAdapter
from jevbench.tasks import load_jsonl
model, rev = sys.argv[1], sys.argv[2]
ad = MlxJevLocalAdapter(endpoint=model, revision=rev, backend="torch", device="cuda", orders=1, repeat=2,
                        pin_prefix="Answer: **", pin_marker="{}")
clf = ad.load()
def single_pass(toks, ids):
    with torch.inference_mode():
        row = clf.model(input_ids=torch.tensor([toks], device="cuda"), logits_to_keep=1, use_cache=False).logits[0, -1].float()
    return torch.softmax(row[torch.tensor(ids, device="cuda")], -1).tolist()
prompts = []
for t in load_jsonl("vendor/jevbench/datasets/public/hard.jsonl")[:20]:
    prompts.append((t.id, ad._tokens(ad._prompt(t, list(range(len(t.labels))))), ad._letters(len(t.labels))))
filler = "The quarterly operations review covered inventory, staffing, vendor contracts and facility maintenance. "
for n in (4000, 12000):
    st = filler * (n // 20)
    t = types.SimpleNamespace(state=st, labels=["a", "b", "c", "d"], task_id="x",
                              question={"type": "choice", "instructions": "Which fits?", "criteria": {k: k for k in "abcd"}})
    prompts.append((f"long-{n}", ad._tokens(ad._prompt(t, list(range(4)))), ad._letters(4)))
worst = {2048: 0.0, 512: 0.0}; flips = {2048: 0, 512: 0}
for name, toks, ids in prompts:
    ref = single_pass(toks, ids)
    for ch in (2048, 512):
        clf.PREFILL_CHUNK = ch
        p, _ = clf.read(toks, ids)
        worst[ch] = max(worst[ch], max(abs(a - b) for a, b in zip(p, ref)))
        flips[ch] += p.index(max(p)) != ref.index(max(ref))
print(f"{model}: {len(prompts)} prompts (up to {max(len(t) for _, t, _ in prompts)} tokens) | "
      f"chunk 2048: max |dP| {worst[2048]:.2e}, decision changes {flips[2048]} | chunk 512: max |dP| {worst[512]:.2e}, decision changes {flips[512]}")
