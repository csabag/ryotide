"""Does RYOTIDE's question echo (S/Q/Q) transfer to the real Jev?

Jev returns probabilities, not logits, so only changes expressible in the REQUEST can
transfer. The echo is one: Jev lays out state, then question; appending the question
(instructions + lettered options) to the end of a TEXT state makes that
state / question / question. JSON-object states are sent unchanged -- turning them into
text would change two things at once -- and serve as a within-run noise control.

    uv run python bench/run_jev_variants.py --variant base --tag jev-base-1
    uv run python bench/run_jev_variants.py --variant echo --tag jev-echo-1

Uses JevBench's own TypeSafeAdapter against OpenRouter; only build_request differs.
The OpenRouter key is read from .env into this process only and never printed.
"""
import argparse, copy, json, os, shutil, sys, time
sys.path.insert(0, "src")
from jevbench.adapters import TypeSafeAdapter
from jevbench.budget import Ledger
from jevbench.runner import Runner
from jevbench.summarize import summarize
from jevbench.tasks import load_jsonl

LETTERS = "ABCDEFGHIJKLMNOP"


def question_block(q: dict) -> str:
    """The question as RYOTIDE lays it out: instructions, then lettered options."""
    crit = q.get("criteria")
    lines = [f"Question: {q['instructions']}"]
    if isinstance(crit, dict):
        items = list(crit.items())
        if q["type"] == "noul":
            items = [("yes", crit.get("true", "")), ("no", crit.get("false", ""))]
        lines.append("Options:")
        lines += [f"{LETTERS[i]}. {k}" + (f" - {v}" if v else "") for i, (k, v) in enumerate(items)]
    elif isinstance(crit, list):
        lines.append("Levels:")
        lines += [f"{i}. {v}" for i, v in enumerate(crit)]
    return "\n".join(lines)


class EchoAdapter(TypeSafeAdapter):
    """Appends the question to text states, so Jev sees state / question / question."""
    def build_request(self, task):
        body = super().build_request(task)
        if isinstance(task.state, str):
            body = copy.deepcopy(body)
            body["state"] = task.state + "\n\n" + question_block(task.question)
        return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("base", "echo"), required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="typesafe/jev-1.13")
    ap.add_argument("--cap-usd", type=float, default=0.50)
    a = ap.parse_args()

    for line in open(".env"):                                   # key stays in this process
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY not found in .env")

    tasks = [t for f in ("original", "easy", "hard") for t in load_jsonl(f"vendor/jevbench/datasets/public/{f}.jsonl")]
    out = f"results/jevbench/{a.tag}"; shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    raw = f"/tmp/jev-raw/{a.tag}"; shutil.rmtree(raw, ignore_errors=True)
    cls = EchoAdapter if a.variant == "echo" else TypeSafeAdapter
    ad = cls(endpoint="https://openrouter.ai/api", model=a.model, key_env="OPENROUTER_API_KEY",
             price_input_per_m=0.042, price_output_per_m=0.0)
    print(f"[jev] variant={a.variant} model={a.model} tasks={len(tasks)}", flush=True)
    runner = Runner(ad, Ledger(f"{out}/ledger.jsonl", cap_usd=a.cap_usd), raw_dir=raw, default_reserve_usd=0.0001)
    t0 = time.time()
    records = runner.run_all(tasks, progress_every=50, results_path=f"{out}/results.jsonl")
    json.dump(summarize(tasks, records), open(f"{out}/summary.json", "w"), indent=1, default=str)
    ok = [r for r in records if r["ok"]]
    print(f"=== {a.tag} === n={len(records)} ok={len(ok)} correct={sum(r['correct'] for r in records)} "
          f"cost ${sum((r.get('cost_usd') or 0) for r in records):.4f} wall {time.time()-t0:.0f}s "
          f"models {sorted({r.get('model') for r in records})}")


if __name__ == "__main__":
    main()
