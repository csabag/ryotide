import sys, types, torch
sys.path.insert(0, "src"); sys.path.insert(0, "vendor/jevbench")
from ryotide.jevbench_adapter import MlxJevLocalAdapter
from jevbench.tasks import load_jsonl
ad = MlxJevLocalAdapter(endpoint=sys.argv[1], revision=sys.argv[2], backend="torch", device="cuda", orders=1, repeat=2,
                        pin_prefix="Answer: **", pin_marker="{}")
clf = ad.load()
def single(toks, ids):
    with torch.inference_mode():
        row = clf.model(input_ids=torch.tensor([toks], device="cuda"), logits_to_keep=1, use_cache=False).logits[0, -1].float()
    return torch.softmax(row[torch.tensor(ids, device="cuda")], -1).tolist()
print(f"{'item':30s} {'tokens':>6s} | {'top-2 margin (single)':>21s} | max|dP| c2048  c512 | decision c2048/c512")
for t in load_jsonl("vendor/jevbench/datasets/public/hard.jsonl")[:20]:
    toks = ad._tokens(ad._prompt(t, list(range(len(t.labels))))); ids = ad._letters(len(t.labels))
    ref = single(toks, ids); s = sorted(ref, reverse=True)
    out = []
    for ch in (2048, 512):
        clf.PREFILL_CHUNK = ch; p, _ = clf.read(toks, ids)
        out.append((max(abs(a - b) for a, b in zip(p, ref)), "same" if p.index(max(p)) == ref.index(max(ref)) else "FLIP"))
    if max(o[0] for o in out) > 0.01 or any(o[1] == "FLIP" for o in out):
        print(f"{t.id:30s} {len(toks):6d} | {s[0]-s[1]:21.3f} | {out[0][0]:.3f}  {out[1][0]:.3f} | {out[0][1]}/{out[1][1]}")
