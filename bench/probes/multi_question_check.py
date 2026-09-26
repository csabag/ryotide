"""run_many (state read once, one branch per question) vs answering each question alone.

    uv run python bench/probes/multi_question_check.py [model] [mlx|torch]
"""
import sys, json, time, types, random
from collections import defaultdict
sys.path.insert(0, "src"); sys.path.insert(0, "vendor/jevbench")
from ryotide.jevbench_adapter import MlxJevLocalAdapter
from jevbench.tasks import load_jsonl
model = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/gemma-4-e4b-it-8bit"
backend = sys.argv[2] if len(sys.argv) > 2 else "mlx"
ad = MlxJevLocalAdapter(endpoint=model, orders=1, repeat=2, pin_prefix="Answer: **", pin_marker="{}", backend=backend)
ad.load()

def compare(groups, label):
    same = n = 0; worst = 0.0; t_many = t_one = 0.0; shared = []
    for tasks in groups:
        t0 = time.perf_counter(); many = ad.run_many(tasks); t_many += time.perf_counter() - t0
        t0 = time.perf_counter(); one = [ad.run(t) for t in tasks]; t_one += time.perf_counter() - t0
        for a, b in zip(many, one):
            assert a.ok and b.ok, (a.error, b.error)
            n += 1; same += a.label == b.label
            worst = max(worst, max(abs(a.probs[k] - b.probs[k]) for k in a.probs))
        shared.append(many[0].raw["runtime"]["shared_prefix_tokens"])
    print(f"{label}: {n} questions in {len(groups)} requests | identical decisions {same}/{n} | "
          f"max |dP| {worst:.2e} | shared prefix ~{sum(shared)//len(shared)} tok | "
          f"time shared {t_many:.1f}s vs one-by-one {t_one:.1f}s", flush=True)

# 1-2: synthetic scenarios, 5 questions about one state each
by = defaultdict(list)
for t in load_jsonl("data/synthetic/echo-v1.jsonl"):
    by[t.id.rsplit("-", 1)[0]].append(t)
compare([v for k, v in sorted(by.items())][:40], "synthetic, 5 questions/state")

# 3: a 40-option coded question sharing its state with small ones (digits reader)
big = load_jsonl("data/synthetic/many-options-mid.jsonl")[30:40]          # 40-option items
mixed = []
for t in big:
    yn = types.SimpleNamespace(id=t.id + "-yn", state=t.state, labels=["no", "yes"], expected="yes",
                               question={"type": "noul", "instructions": "Is this a banking request?",
                                         "criteria": {"true": "It concerns a bank account or service.", "false": "It does not."}})
    mixed.append([t, yn])
compare(mixed, "40-option coded question + yes/no, one state")

# 4: ToolRet-shaped: one ~3k-token state, 100 yes/no questions
rng = random.Random(5)
doc = " ".join(f"Tool {i}: retrieves {rng.choice(['invoices','shipments','tickets','payroll','leads'])} "
               f"for region {rng.randint(1, 40)} using API v{rng.randint(1, 5)}." for i in range(260))
qs = [types.SimpleNamespace(id=f"tool-{i}", state="Catalogue:\n" + doc + "\n\nQuery: find open support tickets for region 7.",
                            labels=["no", "yes"], expected="no",
                            question={"type": "noul", "instructions": f"Is tool {i} relevant to the query?",
                                      "criteria": {"true": "Relevant.", "false": "Not relevant."}}) for i in range(100)]
compare([qs], "ToolRet-shaped, 100 questions on a long state")
