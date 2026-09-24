# Pre-registration: when does the question echo help?

Written before any model was run on `data/synthetic/echo-v1.jsonl`
(generator `bench/synthetic/gen_echo_tasks.py --n 40 --seed 7`; 600 tasks).

## Background

On JevBench's public items the echo (question body emitted twice, S/Q/Q) helped
`choice` items and did not help yes/no (`noul`) items. Every 2-option public item is
`noul`, so option COUNT and question TYPE were never separated. The mechanism we
claim: the second copy of the option list attends to all options, so it helps when
the decision is a comparison between named alternatives.

## Conditions

Models: Gemma 4 E4B (`mlx-community/gemma-4-e4b-it-8bit`) and Qwen3.5-4B
(`mlx-community/Qwen3.5-4B-MLX-8bit`). Read position pinned (`Answer: **`, `{}`),
one option order per item. Echo OFF (`--repeat 1`) vs echo ON for every item
(`--repeat 2 --echo-min-options 0`). Nothing else differs.

## Hypotheses

- **H1** Echo gain on `choice4` exceeds echo gain on `noul` (same scenarios).
- **H2** Echo gain on `choice2` is positive. If it is, the gate should be "echo on
  choice questions" (mechanism). If it is zero or negative, the option-count gate
  (> 2 options) is the better rule.
- **H3** Echo raises the share of scenarios whose `choice4` answer is identical
  across the three option rotations (`choice4`, `choice4r1`, `choice4r2`).

## Analysis

Per form: accuracy with and without echo; items fixed and broken by the echo;
two-sided exact sign test on fixed vs broken, per model and pooled. H3: count of
scenarios with all three rotations agreeing, echo OFF vs ON, McNemar-style sign
test on scenarios that change. alpha = 0.05; with 120 scenarios per form these tests
can only detect effects of several items, and a null is reported as a null.

## What may be adjusted after seeing data

Only task difficulty, and only from echo-OFF accuracy (to avoid ceiling or floor),
before any echo-ON run. If regenerated, the new seed and reason are recorded here.

## Results (added after the runs; nothing above was changed)

Difficulty check (echo OFF only) showed no ceiling or floor (Gemma overall 0.747;
lowest cell lateness/noul 0.53, highest routing/choice2 0.95), so the set was used
as generated -- no regeneration, seed 7 stands. Marker mass >= 0.92 on every run.

| | Gemma 4 E4B | Qwen3.5-4B | pooled fixed / broke |
|---|---|---|---|
| noul | 0.767 -> 0.800 (+4) | 0.617 -> 0.675 (+7) | 22 / 11, p = 0.080 |
| choice2 | 0.883 -> 0.908 (+3) | 0.775 -> 0.767 (-1) | 8 / 6, p = 0.791 |
| choice4, all rotations | 0.694 -> 0.736 (+15) | 0.617 -> 0.625 (+3) | 39 / 21, p = 0.027 |
| rotation agreement (H3) | 88 -> 88 of 120 | 82 -> 82 of 120 | 18 / 18, p = 1.000 |

- **H1 not supported.** The echo helps `noul` about as much as `choice4`.
- **H2 null.** No detectable effect on two-option choice.
- **H3 rejected.** The echo does not change order sensitivity at all.

The echo helps overall (clearly on Gemma, p = 0.004 on choice4; a wash on Qwen), but
not through the mechanism we proposed: an order-invariance effect is the direct
prediction of "the options become mutually visible", and it is exactly zero. The
data fit a simpler account -- a second read of the question after the state, which
would help every question type and leave order sensitivity untouched.

The "> 2 options" gate is not supported by this independent data: here the echo
helps yes/no items. Its only support is the public JevBench items.
