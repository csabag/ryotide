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
