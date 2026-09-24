# JevBench submission — draft

Status: **draft, not posted.** Everything between the two rules below is the issue
text for <https://github.com/fstandhartinger/jevbench/issues>, ready to paste. The
checklist after it is for us.

---

**Title:** `[bench request]: RYOTIDE-Qwen and RYOTIDE-Gemma (frozen Qwen3.5-4B / Gemma 4 E4B, native option-marker probabilities, TypeSafe wire format)`

Hi Florian, I'd like to submit two entries from **RYOTIDE** (Roll Your Own Typed
Inference Decision Engine). Both read typed decisions from a frozen public model in
one forward pass, with no generated tokens, and serve TypeSafe's wire format
(`POST /v1/systemone`), so your unchanged `typesafe` adapter runs them. They share
all code and differ only in the model and one temperature, so I'm filing them
together. Please rank them as two rows, or run whichever you prefer.

## System details

- **Repo:** <https://github.com/csabag/ryotide>, tag **`v0.1.0`** (commit `6bbb615`), MIT
- **Readout:** the state, then the question with lettered options, then the question
  once more (S/Q/Q). The distribution is the next-token softmax over the option
  letters after an assistant-turn prefix `Answer: **`, masked to those letters and
  renormalised, with thinking off. `noul` is read over yes/no options and `score` over
  its levels. `choice` options are put in natural order, so digit runs compare as
  numbers and everything else sorts alphabetically. JSON key order is not meaningful.
- **Output tokens:** 0 per decision. One forward pass per decision, one option order.
- **Training:** none. The weights are unmodified at the pinned revisions. There is no
  adapter, LoRA or trained head.

| entry | weights (revision) | licence | temperature | `/health` prompt hash |
|---|---|---|---|---|
| **RYOTIDE-Qwen** | `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Apache-2.0 | 1.0 | `135c2ebd8537` |
| **RYOTIDE-Gemma** | `google/gemma-4-E4B-it` @ `ee0ef6023621cff504d758262d4e04895a5af4a2` | Apache-2.0 | 1.924 | `1d466fc5a4e1` |

## How to run

One NVIDIA GPU, bf16. Peak memory we measured over a full hard-tier run: Qwen 9.1 GB,
Gemma 16.5 GB. The kernels for Qwen3.5's Gated DeltaNet come from
`flash-linear-attention` and install with the `cuda` extra.

```sh
git clone https://github.com/csabag/ryotide && cd ryotide && git checkout v0.1.0
uv sync --frozen --extra cuda
PYTHONPATH=src uv run python -m ryotide.server --preset ryotide-qwen --port 8778    # or --preset ryotide-gemma
# prints "ryotide ready ..." after loading and one warm-up request; no authentication,
# binds 127.0.0.1 by default. GET /health reports the entry, revision and prompt hash.

python -m jevbench.cli run --adapter typesafe --endpoint http://127.0.0.1:8778 \
  --key-env '' --model ryotide-qwen --tasks <tasks> --results <out> ...
```

A `Dockerfile` is included (`docker run --gpus all ... ryotide --preset ryotide-qwen`)
if you would rather build a container. The weights download at start-up at the pinned
revision.

## Reference run (public items)

Through your harness (vendored at `v1.4.1`) with `--adapter typesafe` against the
server above: one RTX PRO 6000 Blackwell (a 24 GB MIG slice), bf16, serial, from
localhost.

| tier | items | RYOTIDE-Qwen | RYOTIDE-Gemma |
|---|---|---|---|
| easy | 48 | 1.000 | 1.000 |
| standard (original) | 72 | 0.875 | 0.958 |
| hard | 111 | 0.658 | 0.613 |
| **public total** | 231 | **184 (0.797)** | **185 (0.801)** |

Every request returned a valid distribution (231 of 231 for each entry).

- **Hard-tier top-label ECE:** Qwen 0.076, Gemma 0.189.
- **Raw latency:** p50 58 ms and p95 615 ms for Qwen; p50 64 ms and p95 782 ms for
  Gemma. These are measured on our machine, with no network.
- **Input tokens per decision:** mean about 740, median about 240, max about 4.2k. The
  echo repeats the question and options, never the state.

## Cost basis (suggestion)

These are the same hosted-price estimates the board already uses for these weights,
with 0 output tokens:

- **Qwen:** DeepInfra `Qwen/Qwen3.5-4B`, $0.03 per million input tokens.
- **Gemma:** DeepInfra `google/gemma-4-E4B-it`, $0.02 per million input tokens.

## How the configuration was chosen

I'd rather state this plainly:

- **Development was repeatedly measured on the 231 public items.** No held-out, judge
  or sealed item was ever seen.
- **The read position** (`Answer: **`, bare letters) was chosen by *marker mass*: the
  share of the full-vocabulary softmax that lands on the option letters. That choice
  uses no labels.
- **The echo (S/Q/Q)** was measured on public items. It used to apply only to
  questions with more than two options. A pre-registered test on synthetic
  typed decisions found no support for that gate, so the echo now applies to every
  question. On the public items that costs 3 yes/no decisions. The pre-registration
  and its results are in `bench/synthetic/PREREGISTRATION.md`.
- **Natural option order** was fixed before it was measured. It changes one public
  item.
- **The temperature** was fit on that synthetic set (even scenarios) and adopted only
  if it improved calibration on the held-out half. It was adopted for Gemma (held-out
  ECE 0.124 → 0.070) and rejected for Qwen, whose held-out ECE got worse
  (0.029 → 0.084). JevBench items were never used to fit it, and it changes no
  decisions. The fit reports are in `results/calibration/`.
- **Negative results** (layouts, notes, marker swaps, gated reasoning, trajectory
  rules) are kept in the repo with their per-item outputs.

Thanks for running the benchmark. Happy to adjust anything to fit your process.

---

## Before posting (for us, not part of the issue)

- [ ] Re-read the numbers against `results/jevbench/cuda-wire-{qwen,gemma}-final/`.
- [ ] Decide on one issue for both entries or one per entry. Precedent covers both:
      kev and GLiNER went in one round, Open-Jev 2B and 9B together.
- [ ] Optional: build the Docker image once. Its steps are the verified install, but
      the image itself has not been built.
- [ ] Make sure nothing after `v0.1.0` changes behaviour. Any code change means a new
      tag (`v0.1.1`) and new numbers.
- [ ] Post from the `csabag` account. The maintainers' bot logs requests; expect "logged
      for review", then a measurement run.
