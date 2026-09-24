"""Selective prediction: answer when confident, defer when not.

Low confidence does NOT mean the model is wrong in a flippable way -- accuracy
is monotone in confidence on every model measured here, and never dips below
chance, so inverting a low-confidence answer strictly loses. What low confidence
does mark is a decision not worth acting on. Deferring those is the intervention
that pays.

Two scores are available for ranking decisions by trustworthiness:
  confidence -- probability of the top label
  margin     -- top minus runner-up; ignores how the remaining mass is spread

Which one ranks better is an empirical question per model; `curve()` measures it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

Score = Literal["confidence", "margin"]


@dataclass
class Point:
    threshold: float
    coverage: float           # fraction answered
    selective_accuracy: float # accuracy among answered
    n_answered: int
    n_deferred: int


def score_of(probs: dict, how: Score = "confidence") -> float:
    v = sorted(probs.values(), reverse=True)
    if how == "margin":
        return v[0] - (v[1] if len(v) > 1 else 0.0)
    return v[0]


def curve(items: Sequence[tuple[float, bool]]) -> list[Point]:
    """items: (score, correct). Returns one point per achievable coverage."""
    ordered = sorted(items, key=lambda x: -x[0])
    out, correct = [], 0
    for i, (s, ok) in enumerate(ordered, start=1):
        correct += ok
        out.append(Point(threshold=s, coverage=i / len(ordered),
                         selective_accuracy=correct / i,
                         n_answered=i, n_deferred=len(ordered) - i))
    return out


def aurc(points: Sequence[Point]) -> float:
    """Area under the risk-coverage curve. Lower is better; 0 is an oracle."""
    tot = 0.0
    for a, b in zip(points, points[1:]):
        tot += (1 - a.selective_accuracy + 1 - b.selective_accuracy) / 2 * (b.coverage - a.coverage)
    return tot


def at_coverage(points: Sequence[Point], c: float) -> Point:
    return min(points, key=lambda p: abs(p.coverage - c))


def coverage_for_accuracy(points: Sequence[Point], target: float) -> Point | None:
    """Largest coverage whose selective accuracy still meets `target`."""
    ok = [p for p in points if p.selective_accuracy >= target]
    return max(ok, key=lambda p: p.coverage) if ok else None
