"""RESULTS-v1.1.3.md from v113/out/jevbench-v1.1.3-results.json (and SPEND.md for the GPU bill)."""
import json
from pathlib import Path

JOB = Path(__file__).resolve().parents[1]
R = json.loads((JOB / "v113/out/jevbench-v1.1.3-results.json").read_text())
H = "33/33/33 balanced (headline)"
pc = lambda x: "-" if x is None else f"{100*x:.1f} %"
n1 = lambda x: "-" if x is None else f"{x:.1f}"


def usd(s):
    c = s["cost"]
    if c["usd_per_1000"] is None:
        return "no tariff"
    return ("~" if c["kind"] == "estimate" else "") + f"${c['usd_per_1000']:.4f}"


def row(i, s):
    ta = s["capability"]["tier_accuracy"]
    tag = " **(GPU)**" if s.get("round") == "v1.1.3 GPU round" else ""
    return (f"| {i} | {s['display']}{tag} | **{s['main_score']:.1f}** | {n1(s['sensitivity']['60/20/20 accuracy emphasis'])} | "
            f"{n1(s['capability']['score'])} | {n1(s['speed']['score'])} | {n1(s['cost']['score'])} | {pc(ta['easy'])} | "
            f"{pc(ta['standard'])} | {pc(ta['judge'])} | {s['speed']['p50_s']:.2f} s / {s['speed']['p95_s']:.2f} s | {usd(s)} |")


WHY = {"qwen3.8-27b": "v1.0 run stopped at 223 of 242 after three empty completions in a row (unchanged from v1.1.2)"}
ranked = [s for s in R["systems"] if s["ranked"]]
unranked = [s for s in R["systems"] if not s["ranked"]]
gpu = [s for s in R["systems"] if s.get("round") == "v1.1.3 GPU round"]
L = []
L += ["> [!WARNING]", "> **Work in progress — preliminary results, please don't share or cite yet** (v1.2 with a harder tier is being built).", ""]
L += ["# JevBench v1.1.3 - the GPU round", "",
      f"Generated from `results/v1.1.3/jevbench-v1.1.3-results.json` ({R['generated_utc'][:16].replace('T', ' ')} UTC).", "",
      "**What v1.1.3 is:** the frozen v1.1 task set (314 typed decisions: 72 easy / 96 standard / 146 judge) and the v1.1.2 scoring, "
      "unchanged, plus open Jev rebuilds that need a GPU. We rented GPUs on RunPod and ran each rebuild the way its author "
      "serves it. Every v1.1.2 row is copied unchanged, so only the ranks move. The v1.1.2 numbers stay at tag `v1.1.2`.", "",
      "![Main Score](results/v1.1.3/charts/main-score.png)", "",
      "## Main Score (Balanced 33:33:33)", "",
      "| # | System | Main | 60:20:20 | Capability | Speed | Cost | Easy | Standard | Judge | p50 / p95 | $ per 1,000 decisions |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for s in sorted(ranked, key=lambda s: s["rank_under"][H]):
    L.append(row(s["rank_under"][H], s))
L += ["", "`~` = estimate (no tariff for us): priced like a large inference provider hosting the same weights or size class, "
      "never the GPU rental we paid. **(GPU)** = new in v1.1.3. A `*` in a name marks an adapter mode added after the frozen run, shown beside it.", ""]
if unranked:
    L += ["**Shown, not ranked** (a tier was not attempted in full):", "",
          "| System | Main | Coverage (easy / standard / judge) | Why |", "|---|---|---|---|"]
    for s in unranked:
        cv = s["coverage"]
        L.append(f"| {s['display']} | {n1(s['main_score'])} | {pc(cv['easy'])} / {pc(cv['standard'])} / {pc(cv['judge'])} | {s.get('note') or WHY.get(s['key'], '')} |")
    L.append("")
L += ["## How the GPU entrants were run", "",
      "| System | Author's serving path we reproduced | GPU (RunPod) | Speed measured | GPU-side p50 / p95 | Same answers, GPU-side vs network run |",
      "|---|---|---|---|---|---|"]
for s in gpu:
    g = s["speed"].get("gpu_side") or {}
    gs = f"{g['p50_s']:.3f} s / {g['p95_s']:.3f} s" if g else "-"
    same = f"{g['same_answer_as_network_run']} / {g['of']}" if g else "-"
    L.append(f"| {s['display']} | {s.get('serving') or ''} | {s['speed']['hardware']} | {s['speed']['measured_where']} | {gs} | {same} |")
L += ["",
      "- **Same items, same scorer, same rules as v1.1.** Serial, no retries, immutable run folders, no label changes after seeing predictions. "
      "The answer key never leaves our machine: library-only entrants get the task without its gold label.",
      "- **Speed is network-included, like every remote entrant in v1.1:** requests go from our Hetzner server in Germany over the internet to "
      "the rented GPU (EU data centres, TCP round trip about 55 ms to Romania). The GPU-side column is the same 242 decisions with the client "
      "on the GPU machine: it shows how much of each number is the network (for OpenJev, SemIf and open-alternative-jev that run was on an earlier pod with the same GPU model). API entrants (Jev, Gemini, GPT, DeepSeek, the Modal endpoints) "
      "pay their own, usually longer, round trip; a self-hosted model can sit next to its caller, so this is a real advantage of self-hosting, "
      "and it is also why these Speed scores are not a like-for-like comparison of model compute.",
      "- **Models are loaded before timing starts**, as a running server would be. v1.1's CPU entrants counted their load in the first decision (1 of 242, "
      "so p50/p95 barely move).",
      "- **Cost** follows v1.1.2 exactly: hosted-provider list price of the same weights (or the nearest larger sibling), times measured input "
      "tokens, plus one output token. What we actually paid RunPod is in SPEND below and is not the Cost score.", ""]
L += ["## Findings", "",
      "- **Self-hosted open rebuilds take the top of the Balanced ranking; Jev drops from #2 to #8.** The reason is Speed and Cost, "
      "not accuracy. A 4-9B model on a GPU in a nearby data centre answers in about 0.2 s from Germany, against 0.65 s for Jev's API, and "
      "costs about $0.01-0.05 per 1,000 decisions at hosted-provider prices. On Capability, Jev (97.8) and SemIf (97.7) are level; "
      "only GPT-5.6 Luna is higher (98.2). With the accuracy-heavy 60:20:20 weights, SemIf still leads (89.1) and Jev is #4 (83.3).",
      "- **SemIf (Qwen3.5-4B, frozen, no training) is the strongest rebuild:** 97.9 % standard, 95.2 % judge, #1 under five of six weightings.",
      "- **Option order matters a lot for small frozen models.** open-alternative-jev reads the answer as a letter. Our v1.1 mapping listed "
      "yes/no questions as `A. no, B. yes`; on the 68 answer-judging items it then said \"no\" 61 times where 61 of 68 gold answers are \"yes\" "
      "(20.6 % on that cohort). With the author's own `yes_no()` order (`A. yes, B. no`) the same model scores 72.1 % there. The frozen run is the "
      "official row; the author-order run is shown beside it with a `*`, as v1.1 did for Needle 3's tools mode.",
      "- **Bespoke Nimble 9B refuses prompts over its trained 2,048-token limit** (HTTP 422). On this task set that costs it 2 long answer-judging items (counted wrong, as for everyone).",
      "- **OpenJev (DiffusionGemma) is not fully deterministic:** 237 of 242 answers matched between two runs. All other GPU entrants matched 242/242.",
      "- **A preview of why v1.2 needs a hard tier:** on the 220-item hard tier being frozen for v1.2, these entrants score 44-66 % "
      "(OpenJev 65.5, SemIf 59.5, open-alternative-jev 55.0 / 56.8*, system-one 50.0, Nimble 43.6 with 69 items over its token limit), "
      "against Jev 74.1 % and GPT-5.6 Luna 94.5 % (the v1.2 job's runs). The near-100 % ceiling here hides real gaps. Those runs belong to v1.2 and are handed to that job.",
      ""]
L += ["## Sensitivity (rank under each weighting)", "", "| System | " + " | ".join(R["sensitivity_order"]) + " |",
      "|---|" + "---|" * len(R["sensitivity_order"])]
for s in sorted(ranked, key=lambda s: s["rank_under"][H]):
    L.append(f"| {s['display']} | " + " | ".join(f"{s['rank_under'][k]} ({s['sensitivity'][k]:.1f})" for k in R["sensitivity_order"]) + " |")
L += ["", "## Scoring (unchanged from v1.1.2)", ""]
for k in ("capability", "calibration", "speed", "cost", "main", "ranked"):
    L.append(f"- **{k.capitalize()}.** {R['scoring'][k]}")
L += ["", "## Still not measured, and why", "",
      "- **Dasein Labs open-jev**: MLX on Apple Silicon only; no Linux path.",
      "- **JoshuaSP open-jev** (DiffusionGemma 26B-A4B, BF16): the repository ships bounded Modal batch jobs (`modal run infer.py`), not a decision service, and needs an 80 GB GPU. Not run in this round.",
      "- **mini-jev** (r-ms): a preregistered study and teaching bench of letter-reading on Qwen3-4B, not a decision service.",
      "- **system-one-gemma** (Akash Kamat): the base model, Gemma 3 270M, is gated behind Google's licence click-through; we do not accept binding terms on Florian's behalf.",
      "- **jevlike, AlexWortega/openjev, GLiNER2, Needle-style routers**: unchanged from v1.1 (no general text-decision checkpoint, or no distribution over an arbitrary label set without an assumption we would be measuring instead).", ""]
spend = (JOB / "SPEND.md").read_text()
L += ["## GPU bill for this round", "", "RunPod, one pod at a time, hard cap $16. This is what we paid to measure, not the Cost score.", "",
      spend.split("\n\n", 2)[-1].strip(), ""]
L += ["## Files", "", "- `results/v1.1.3/jevbench-v1.1.3-results.json` - the artifact (aggregates only).",
      "- `results/v1.1.3/charts/` - main score, sub-benchmarks, tiers, sensitivity.",
      "- `jevbench/adapters/semif_direct.py`, `so1_decider.py`, `sg_system_one.py`, `remote_inproc.py` - the new adapters; the two server entrants (OpenJev, Nimble) use the existing `typesafe` adapter against the author's own `/v1/systemone`.",
      "- `jevbench/runner.py` - one change: an HTTP 422 (the system refusing an input, e.g. over its context limit) no longer counts toward the three-consecutive-errors stop rule; it still counts as a wrong answer.",
      "- `scripts/v1.1.3/` - the pod runner, the thin transport, the aggregator and the charts."]
(JOB / "v113/out/RESULTS-v1.1.3.md").write_text("\n".join(L) + "\n")
print("ok", len(L))
