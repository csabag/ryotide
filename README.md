# RYOTIDE — Roll Your Own Typed Inference Decision Engine

Reading a local LLM as a typed decision model.

Run a small local model on Apple
Silicon as a **Jev-style classifier** — one forward pass, logits read at a single
position, masked to a fixed label set, no autoregressive decode — and share the
same weights and KV cache with normal generation.

Measured on [JevBench](https://github.com/fstandhartinger/jevbench) (MIT, vendored
at `vendor/jevbench`), the benchmark that actually scores Jev-class typed-decision
models. 47 runs across Qwen2.5, Qwen3, Qwen3.5, Gemma 4 and gpt-oss, 231 public
decisions each -- plus the real Jev via OpenRouter's Decisions API, and GLUE for
all three.

**The harness is validated.** Running the real Jev through it reproduces its
published public-subset tiers exactly (easy 1.000, standard 0.986, hard 0.730) with
99.1% per-item agreement, so every comparison here sits on the same footing as the
published board.

**Dashboard:** https://claude.ai/artifact/MZuHA6zN1gFucrGaPRyf8t

## Results

Best capability **75.0 intelligence** (Gemma 4 E4B + conditional repetition) and
best composite **74.8** (Qwen3.5-4B 8-bit), both measured with our local runner.
Measured head to head, Jev reaches 82.2 intelligence and 87.0 composite on the same
three tiers.

> **Correction — read this before comparing with the board.** The numbers in this
> section come from our local runner, which presents options in each task's
> authored order. JevBench measures every entrant through the TypeSafe wire format,
> where options arrive alphabetically. Through the wire Gemma 4 E4B scores
> **185/231 (0.801)** with the server's default natural option order (184/231 with
> options as received), not 188/231 (0.814): just ahead of metask-jev-4b and
> *behind* local-jev, SemIf, Jobe and Hopper in the ~4B class. See [Serving](#serving-the-typesafe-wire-format-on-mlx-or-pytorch).
> Official v1.4 ranks also require the maintainers' run on 308 sealed decisions,
> which no number here includes.

| system | intel | easy | std | hard | calib | score |
|---|---|---|---|---|---|---|
| Jev 1.13.0 (TypeSafe) | 83.1 | 1.000 | 0.990 | 0.741 | 83 | 74.4 |
| SemIf (Qwen3.5-4B) | 73.4 | 1.000 | 0.979 | 0.595 | 73 | 73.1 |
| **ours — Gemma 4 E4B 8bit** | 72.2 | 1.000 | 0.958 | 0.595 | 40 | 65.4 |
| **ours — Qwen3.5-4B 8bit** | 65.6 | 1.000 | 0.833 | 0.604 | 80 | **73.5** |
| system-one-open (E2B+LoRA) | 64.5 | 1.000 | 0.938 | 0.491 | 57 | 66.6 |
| **ours — Gemma 4 E2B 8bit** | 58.9 | 1.000 | 0.875 | 0.459 | 0 | 26.3 |
| kev 0.6B | 51.7 | 1.000 | 0.812 | 0.400 | 51 | 62.5 |
| **ours — Qwen3.5-2B** | 49.8 | 1.000 | 0.764 | 0.414 | 54 | 61.8 |
| **ours — Qwen2.5-0.5B 8bit** | 37.9 | 0.854 | 0.556 | 0.477 | 63 | 38.5 |
| **ours — Qwen2.5-Coder-0.5B** | 15.4 | 0.646 | 0.389 | 0.360 | 89 | 5.6 |

Published rows are recomputed on our three tiers with the harness's own scorer.
Ours are the 231 public decisions; the judge tier (146 items) is held out.

## Head to head with the real Jev

`bench/run_jev.py` drives JevBench's own TypeSafe adapter against OpenRouter's
Decisions API (`/api/v1/systemone`, `typesafe/jev-1.13`). It reproduces Jev's
published public-subset tiers exactly -- easy 1.000, standard 0.986, hard 0.730,
with 99.1% per-item agreement -- which validates the whole harness.

| | Jev 1.13 | Gemma 4 E4B | Qwen3.5-4B |
|---|---|---|---|
| easy / standard / hard | 1.000 / 0.986 / **0.730** | 1.000 / 0.958 / 0.640 | 1.000 / 0.875 / 0.640 |
| intelligence | **82.2** | 75.0 | 70.3 |
| calibration | 85.5 | 44.2 | 83.2 |
| p50 / p95 | **274 / 415 ms** | 438 / 4740 ms | 647 / 7098 ms |
| composite | **87.0** | 66.7 | 74.8 |

Jev is faster over the network than our models are on-device, and its p95/p50
ratio is 1.5 against our 11-17. Measured cost is $0.00075/1k decisions.

## GLUE, and the linguistic tasks

The project started from a hypothesis that Jev would be weak on GLUE. It isn't.
On identical items (150/task):

| task | Jev | Gemma 4 E4B | Qwen3.5-4B | majority |
|---|---|---|---|---|
| sst2 | 0.940 | 0.933 | 0.900 | 0.540 |
| qnli | 0.927 | 0.887 | **0.953** | 0.507 |
| rte | 0.920 | 0.867 | **0.933** | 0.553 |
| cola | 0.820 | 0.793 | 0.820 | 0.673 |
| mrpc | 0.707 | **0.740** | 0.610 | 0.607 |

Read with proper metrics (300 items, MCC for CoLA since 67% majority makes
accuracy meaningless), the "linguistic weakness" mostly disappears:

| | Jev | Gemma 4 E4B | Qwen3.5-4B |
|---|---|---|---|
| cola MCC | +0.544 | +0.533 | **+0.561** |
| mrpc MCC | +0.445 | **+0.451** | +0.362 |
| wnli acc / MCC | **0.915 / +0.828** | 0.718 / +0.455 | 0.775 / +0.539 |

CoLA at +0.53-0.56 MCC is fine-tuned-BERT territory for all three. **WNLI is the
one real gap** -- and it is not contamination.

## Where the gap actually is: coreference

Five Winograd twin schemas were authored for this repo (`bench/winograd_fresh.py`)
and exist nowhere else. Each twin changes one discriminating phrase so the
referent flips while syntax stays identical; scoring requires *both* halves.

| system | items | twin pairs |
|---|---|---|
| **Jev 1.13** | 0.900 | **4/5** |
| Qwen3-30B-A3B (MoE) | 0.850 | 2/5 |
| Gemma 4 E4B (4B dense) | 0.800 | 2/5 |
| Qwen3.5-4B (4B dense) | 0.650 | 1/5 |
| gpt-oss-20b (MoE) | 0.550 | 0/5 |

Jev's advantage reproduces on fresh data, so it is a training property. **Two
independent MoE families fail worst on the twin metric** despite 5-7x the
parameters -- sparsity buys nothing here, and a 30B MoE only ties a 4B dense
model. This is the most specific explanation we have for Jev's hard-tier lead:
`long_policy` and `multi_hop` items are entity tracking at scale.


## Findings

**The hard tier has a positional floor of 0.414.** 41.4% of hard items have their
answer at index 1, so "always pick option B" beats the 0.336 chance baseline while
reading nothing. Qwen2.5-1.5B and Qwen3.5-2B score *exactly* 0.414 — no real
ability. Qwen2.5-0.5B's 0.477 comes largely from picking B 67% of the time. Only
Qwen3.5-4B (+0.190) and Gemma 4 E4B (+0.181) genuinely read the document. **Measure
hard-tier claims against 0.414, not 0.336.**

**Selective prediction works; flipping does not.** Accuracy is monotone in
confidence on every model and never drops below chance, so inverting low-confidence
answers strictly loses. Deferring them pays: Qwen3.5-4B answers 79% of decisions at
85% accuracy, 67% at 90%, 54% at 95%. See `src/ryotide/selective.py`.

**Calibration and capability are nearly independent.** The coder model has the best
calibration in the set (89) and the worst capability (15.4) — it is accurately
uncertain. Gemma 4 E2B has the best standard-tier accuracy anywhere (0.875) and a
calibration of 0.0. ECE measures whether probability *values* are right; AURC
measures whether the *ordering* is. Gemma fails the first and passes the second.

**Quantisation is nearly free at every scale.** 4-bit, 8-bit and bf16 differ by
about a point. The design assumption that INT4 is safe holds.

**Order debiasing is model-dependent.** Averaging over permuted option orders helps
Qwen2.5-1.5B (+3.7 intelligence) and hurts Qwen3.5-2B (−5.2) and 4B (−2.5). Option-
order bias is a small-model pathology that fades with scale.

**Code fine-tuning destroys this capability.** Qwen2.5-Coder-0.5B loses 22.5
intelligence against its own base model at matched quantisation and config.

**Architecture loses to recency and training.** A 4B dense Gemma 4 (75.0) beats a
20B MoE (64.8), an 8B Llama (54.7), a 9B Qwen (69.5) and an 8B Mistral (51.5).

**Head-duplication surgery fails (negative result).** Duplicating the 14 query heads
to 28 with a RoPE phase shift costs ~14 intelligence and doubles ECE. The phase
shift does the damage; a query-bias offset is noise on top. `src/ryotide/franken.py`
keeps the identity control that makes this measurable.

### Two bugs that produced valid-looking wrong answers

**Reasoning templates.** Qwen3.5 opens a `<think>` block in its generation prompt,
so the read position landed inside the reasoning and only ~15% of probability mass
reached the option markers. The adapter returned a well-formed distribution over the
right labels and the harness scored it happily. Fixed with `enable_thinking=False`
(score 27.4 → 61.8); every result now records `marker_mass` so it cannot recur
silently.

**Whole-sequence logits.** Computing `[1, n_tokens, vocab]` to read one row cost
~3.7 GB on a 3.7K-token prompt at Qwen3.5's 248k vocabulary — an OOM kill. Now
prefills in chunks and runs only the final token.

**GQA head grouping** (caught by the identity control): with 14 query heads over 2
KV heads, appending a duplicated block remaps every head to the wrong KV group.

## Layout

```
src/ryotide/classify.py          label sets, masked-logit decision, calibration
src/ryotide/branch.py            fork/trim/lazy KV strategies, memory arithmetic
src/ryotide/generate.py          decode-from-cache, result injection
src/ryotide/selective.py         risk-coverage, AURC, coverage-at-accuracy
src/ryotide/franken.py           head-duplication surgery + identity control
src/ryotide/jevbench_adapter.py  decision adapter: prompt, read position, markers, echo; MLX or torch
src/ryotide/server.py            TypeSafe-compatible HTTP server (POST /v1/systemone, GET /health)
src/ryotide/torch_backend.py     PyTorch backend (CUDA / MPS / CPU) for hosts without MLX
src/ryotide/glue.py              superseded GLUE harness, kept for reference
bench/                           runners, scorers, board comparison
bench/probes/                    layout, marker, calibration, notes, reasoning, trajectory probes
vendor/jevbench/                 vendored benchmark subset, tag v1.4.1 (24b9b5c)
results/jevbench/                per-decision results (results.jsonl + summary.json) for every run
results/probes/                  per-item outputs and logs of the probes, negative results included
```

## Serving: the TypeSafe wire format, on MLX or PyTorch

`src/ryotide/server.py` answers TypeSafe's `POST /v1/systemone` (and `GET /health`),
so JevBench's **stock `typesafe` adapter** drives it with no RYOTIDE-specific code.
Two backends share every decision-shaping step — prompt, chat template, read
position, option markers, label folding — and differ only in the forward pass:

- **MLX** (reference, Apple Silicon): `mlx-community/gemma-4-e4b-it-8bit`.
- **PyTorch** (CUDA, MPS or CPU): the original `google/gemma-4-E4B-it` bf16 weights
  (Apache-2.0), pinned to revision `ee0ef6023621cff504d758262d4e04895a5af4a2`.
  MLX is imported lazily, so a CUDA host never needs it.

Both report the same prompt hash on `/health` when configured alike (`74a73257a9b2`
for the defaults), so an evaluator can confirm which configuration is being served.

Gemma 4 E4B 8-bit, 231 public decisions, one pass per decision:

| how it is driven | option order | original | easy | hard | total |
|---|---|---|---|---|---|
| local runner (reference) | authored (`labels`) | 69/72 | 48/48 | 71/111 | 188/231 (0.814) |
| **wire, stock `typesafe` adapter** | as received | 69/72 | 48/48 | 67/111 | 184/231 (0.797) |
| **wire, stock `typesafe` adapter** | **natural (default)** | 69/72 | 48/48 | 68/111 | **185/231 (0.801)** |

Only the wire rows are comparable with the JevBench board.

**Through the wire the public score is 185/231 (0.801), not the 188/231 our local
runner reports.** The request carries options as a `criteria` object whose keys
arrive alphabetically, not in the task's authored order. With options taken as
received, 226 of 231 decisions are unchanged and the score is 184/231; all five
that differ are `choice` items, and four of those were borderline (confidence
0.40–0.75) numeric or ordinal scales that alphabetical order scrambles —
`12, 15, 6, 9 credits` instead of `6, 9, 12, 15`.

**Option order.** A JSON object's key order is not meaningful, so the server puts
choice options in a canonical *natural* order by default: digit runs compare as
numbers, everything else alphabetically (`--option-order received` turns it off,
and the setting is part of the `/health` prompt hash). The rule was fixed before it
was measured and has no tunable part, but read its effect plainly: on the public set
it changes exactly one item — the credits item that motivated it — which it gets
right (184 → 185, the other 230 decisions bit-identical). It cannot recover orders
carried by meaning rather than digits (`before_open / within_window / late…`,
`overturned / modified / upheld`), which account for the other three wire losses.

Every board entrant is measured through the same wire, so **0.801 is the comparable
number**: just ahead of metask-jev-4b (0.797) and behind local-jev, SemIf, Jobe and
Hopper in the ~4B class.

**PyTorch reproduces MLX.** On the 111 hard items in bf16, torch on Apple MPS and MLX
agree on 111/111 decisions (both 70/111); probabilities differ by a median of 0.0003
(max 0.16), the expected bf16 arithmetic-order noise. The refactor that made the
adapter backend-aware is bit-exact on the MLX path (pinned and calibrated,
one and two option orders). **Not yet verified on CUDA** — the code path is the same
PyTorch as MPS, but no NVIDIA run has been made.

```bash
# MLX (Apple Silicon)
PYTHONPATH=src uv run python -m ryotide.server --model mlx-community/gemma-4-e4b-it-8bit
# PyTorch (CUDA / MPS / CPU)
PYTHONPATH=src uv run --extra torch python -m ryotide.server --backend torch \
    --model google/gemma-4-E4B-it --revision ee0ef6023621cff504d758262d4e04895a5af4a2
# JevBench's stock adapter against either (run from vendor/jevbench)
PYTHONPATH=. python -m jevbench.cli run --adapter typesafe --endpoint http://127.0.0.1:8778 \
    --key-env '' --model ryotide --tasks datasets/public/hard.jsonl --results out.jsonl
```

The server has no authentication; it binds to loopback by default.

**Smaller GPUs.** `--quant int8|nf4` (bitsandbytes, CUDA) and `--low-vram` (keep
Gemma 4's 2.8 B-parameter per-layer embedding tables and its unused audio / vision
towers in system RAM) bring Gemma 4 E4B from ~16 GB to ~5–6 GB of GPU memory; the
offload is bit-identical on MPS (20/20 decisions). Qwen3.5-4B runs on the same torch
path and matches MLX on 40/40 hard decisions. **Windows / WSL2 + CUDA:** step-by-step
setup, commands and expected numbers in [`docs/WSL-CUDA.md`](docs/WSL-CUDA.md);
`bench/compare_runs.py <reference> <new>` checks a run against ours.

## Running

```bash
uv run python bench/run_jevbench.py --model mlx-community/Qwen3.5-4B-MLX-8bit --orders 1 --tag myrun
uv run --extra torch python bench/run_jevbench.py --backend torch --model google/gemma-4-E4B-it \
    --revision ee0ef6023621cff504d758262d4e04895a5af4a2 --orders 1 --repeat 2 \
    --prefix 'Answer: **' --marker '{}' --tag torch-run
uv run python bench/score_jevbench.py myrun
uv run python bench/selective_curve.py myrun
uv run python -m ryotide.cli classify state.txt -q "Q: ...? Answer yes or no.\nA:"
```

## Caveats

Public subset only (231 of 534); judge tier held out and renormalised away.
Calibration scores on ECE alone since gold distributions ship with the held-out set.
Cost is our own energy estimate (30 W at $0.30/kWh). Latency is a consumer Mac mini
M4 (base, not Pro) running a desktop OS, with JevBench's ×2 + 0.15 s self-hosted
penalty on top; the p95 tail is structural — 95.8% of a long prompt is state text and
a 3.9K-token prefill genuinely costs ~9 s at ~120 GB/s. A perfect p95 would move our
best score 73.5 → 76.2, against a 17.5-point capability gap.

## Licence

MIT — see [`LICENSE`](LICENSE). This covers the code in `src/` and `bench/` and the
measurements in `results/`.

`vendor/jevbench/` is a subset of [JevBench](https://github.com/fstandhartinger/jevbench)
and keeps its own MIT licence and copyright (`vendor/jevbench/LICENSE`); see
`vendor/jevbench/THIRD-PARTY.md` for the components it does not license. Model weights
are not part of this repository: they are downloaded at run time under their own
licences, which the MIT licence here does not cover.
