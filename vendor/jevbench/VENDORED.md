# Vendored: JevBench (subset)

Pinned so the benchmark survives upstream changes and runs offline.

| | |
|---|---|
| upstream | https://github.com/fstandhartinger/jevbench |
| tag | `v1.4.1` |
| commit | `24b9b5c1609a7a9e8fa14f49e5985a836c9dc842` |
| dated | 2026-09-23 19:55:01 +0000 |
| subject | Release JevBench v1.4.1 additions |
| licence | MIT (harness and the public decisions used here; see `LICENSE`, `THIRD-PARTY.md`) |

`v1.4.1` is one commit on top of `v1.4.0` (the v1.4 scoring release) and adds
entrants only; at vendoring time it was tagged but not yet merged into `main`.

## What is included

The harness (`jevbench/`), `datasets/` (public decisions + `manifest.json`),
`docs/`, `scripts/`, `tests/`, the licence and method files, and only the
upstream results this project reads (`results/v1.2/jevbench-v1.2-results.json`,
the `results/v1.4/` and `results/v1.4.1/` aggregates) plus the three v1.2 files
upstream's own test suite needs (`jevbench-v1.2-per-task.json`,
`jevbench-v1.2-topics.json`, `cost-correction-v1.2.3.json`). With those, the
upstream suite passes here exactly as on a full clone: 104 passed, 5 subtests.

## What is deliberately left out

Upstream charts (`*.png`), the v1.1 / v1.1.3 result archives, the 4 MB v1.2
combination dump and the other v1.2 run artifacts, and the `.git` directory
(4.3 MB vendored instead of ~17 MB). Nothing inside
an included file was modified. To restore the full tree, clone upstream at the
commit above.

## Integrity

The three public decision files are byte-identical to the v1.3.0 copy this
project previously vendored, so every earlier measurement here remains valid.
Their SHA-256 hashes match upstream's `datasets/manifest.json`:

| file | sha256 (prefix) |
|---|---|
| `datasets/public/original.jsonl` | `5c2414edb3006b8b` |
| `datasets/public/easy.jsonl` | `231df3c2c8e88a1a` |
| `datasets/public/hard.jsonl` | `89e9e6becb33ed88` |

## Scoring note

v1.4 ranks only systems with a completed measurement on 308 **sealed** decisions,
run by the maintainers. Scores computed here from public items alone are
v1.3-style projections, not official v1.4 scores.

Our adapter is not in this tree; it lives in `src/ryotide/` so this directory
stays an unmodified subset of upstream.
