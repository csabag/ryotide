"""Progressive-refinement probe. Per item, two caches:
  decision : S/Q/Q, the usual masked read (baseline) -- full probability vector saved
  trajectory: HEAD + Q-with-options + state streamed in 64-token (word-snapped) chunks;
              at p0 (before any state) and after every chunk: fork, close the user turn
              with the model's own template tail + 'Answer: **', masked read, discard fork.
Saves everything; combination rules are swept offline. usage: traj_probe.py MODEL TAG"""
import sys, json, random, time
sys.path.insert(0, "src")
import mlx.core as mx
from ryotide.jevbench_adapter import MlxJevLocalAdapter, _state_text
from ryotide.branch import fork_cache
from jevbench.tasks import load_jsonl

MODEL, TAG = sys.argv[1:3]
CHUNK_TOK = 64
tasks = random.Random(1).sample(load_jsonl("vendor/jevbench/datasets/public/hard.jsonl"), 40)
ad = MlxJevLocalAdapter(endpoint=MODEL, orders=1, repeat=2, pin_prefix="Answer: **", pin_marker="{}")
clf = ad.load(); tok = clf.tokenizer
enc = lambda s: tok.encode(s, add_special_tokens=False)
try:
    tmpl = tok.apply_chat_template([{"role": "user", "content": "XSENTINELX"}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=False)
except (TypeError, ValueError):
    tmpl = tok.apply_chat_template([{"role": "user", "content": "XSENTINELX"}], add_generation_prompt=True, tokenize=False)
HEAD, TAIL = tmpl.split("XSENTINELX")
TAIL_IDS = enc(TAIL + "Answer: **")
print(f"model={MODEL}  HEAD={HEAD!r}  TAIL={TAIL!r}", flush=True)

def chunks(ids):
    out, i = [], 0
    while i < len(ids):
        j = min(len(ids), i + CHUNK_TOK); k = j
        while k < len(ids) and k - j < 16 and not tok.decode([ids[k]])[:1].isspace(): k += 1
        j = k if (k < len(ids) and tok.decode([ids[k]])[:1].isspace()) else j
        out.append(ids[i:j]); i = j
    return out

def probs(row, k):
    sel = mx.array(ad._letters(k, "{}"))
    mass = float(mx.sum(mx.softmax(row.astype(mx.float32))[sel]).item())
    return mx.softmax(row[sel].astype(mx.float32)).tolist(), mass

qonly = MlxJevLocalAdapter(endpoint=MODEL, orders=1, repeat=1); qonly._clf = clf   # question body, ONE copy
out = open(f"results/probes/traj_{TAG}_items.jsonl", "w"); mism = 0
for n, t in enumerate(tasks, 1):
    k = len(t.labels); gold = t.labels.index(str(t.expected)); order = list(range(k))
    state = _state_text(t.state)
    t0 = time.perf_counter()
    base_p, base_m = probs(clf.logits_at_end(clf.prefill([]), ad._tokens(ad._prompt(t, order), suffix="Answer: **")), k)
    tb = time.perf_counter() - t0

    full = qonly._prompt(t, order); assert full.startswith(state)
    qbody = full[len(state):].lstrip("\n")
    head_ids, q_ids, st_ids = enc(HEAD), enc(qbody + "\n\n"), enc(state)
    ref = ad._tokens(qbody + "\n\n" + state, suffix="Answer: **")
    if head_ids + q_ids + st_ids + TAIL_IDS != ref: mism += 1          # tokenization sanity

    t0 = time.perf_counter()
    cache = clf.prefill(head_ids); clf.logits_at_end(cache, q_ids)
    traj, masses = [], []
    def checkpoint():
        br = fork_cache(cache); p, m = probs(clf.logits_at_end(br, TAIL_IDS), k); del br
        traj.append(p); masses.append(m)
    checkpoint()                                                        # p0: options, no evidence
    for c in chunks(st_ids):
        clf.logits_at_end(cache, c); checkpoint()
    del cache
    tt = time.perf_counter() - t0

    am = lambda p: max(range(k), key=lambda j: p[j])
    flips = sum(am(traj[i]) != am(traj[i-1]) for i in range(1, len(traj)))
    out.write(json.dumps(dict(n=n, k=k, gold=gold, base=base_p, base_mass=base_m, traj=traj, traj_mass=masses,
                              state_tok=len(st_ids), base_s=tb, traj_s=tt)) + "\n"); out.flush()
    print(f"[{n:2d}/40] tok {len(st_ids):5d} ckpts {len(traj):3d} | base {'ok' if am(base_p)==gold else 'x '}"
          f" p0 {'ok' if am(traj[0])==gold else 'x '} traj-final {'ok' if am(traj[-1])==gold else 'x '}"
          f" flips {flips:2d} min-mass {min(masses):.3f} {tt:5.1f}s", flush=True)
print(f"\ntokenization mismatches vs adapter templating: {mism}/40", flush=True)
