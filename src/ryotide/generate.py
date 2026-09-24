"""Mode 2: normal generation, continuing from a classified branch.

Spec step 5. The point is that classification and generation share weights *and*
state: after classify() the branch cache already holds the 6K-token state plus
the question, so continuing costs only the injected result plus the decode.

Injection format (json vs prose) is a provisional choice per the spec -- it was
never A/B tested. `inject()` takes the format as an argument and `ab_injection()`
exists so flipping it is a measurement, not an argument.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import mlx.core as mx

from .classify import Classifier, Decision


@dataclass
class GenResult:
    text: str
    n_tokens: int
    prefill_tokens: int
    decode_seconds: float
    inject_seconds: float

    @property
    def decode_tps(self) -> float:
        return self.n_tokens / self.decode_seconds if self.decode_seconds else 0.0


def format_result(decision: Decision, fmt: str = "json") -> str:
    if fmt == "json":
        import json
        return f"\n{json.dumps(decision.as_json())}\n"
    if fmt == "prose":
        conf = decision.probs[decision.label]
        return (f"\nThe answer is {decision.label} "
                f"(confidence {conf:.0%}).\n")
    if fmt == "bare":
        return f"\n{decision.label}\n"
    raise ValueError(f"unknown format {fmt!r}")


def generate_from(
    clf: Classifier,
    cache: list,
    prompt_tokens: Sequence[int],
    max_tokens: int = 128,
    temp: float = 0.0,
) -> GenResult:
    """Decode from an existing cache. The cache is extended in place."""
    model, tok = clf.model, clf.tokenizer
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])

    t0 = time.time()
    ids = mx.array(list(prompt_tokens))[None]
    logits = model(ids, cache=cache)[:, -1, :]
    mx.eval(logits)
    inject_s = time.time() - t0

    out: list[int] = []
    t1 = time.time()
    for _ in range(max_tokens):
        if temp <= 0:
            nxt = int(mx.argmax(logits, axis=-1).item())
        else:
            nxt = int(mx.random.categorical(logits * (1 / temp)).item())
        if nxt in eos:
            break
        out.append(nxt)
        logits = model(mx.array([[nxt]]), cache=cache)[:, -1, :]
        mx.eval(logits)
    decode_s = time.time() - t1

    return GenResult(
        text=tok.decode(out),
        n_tokens=len(out),
        prefill_tokens=len(prompt_tokens),
        decode_seconds=decode_s,
        inject_seconds=inject_s,
    )


def classify_then_generate(
    clf: Classifier,
    branch_cache: list,
    decision: Decision,
    follow_up: str,
    fmt: str = "json",
    max_tokens: int = 128,
    temp: float = 0.0,
) -> GenResult:
    """The hybrid: feed the label back into the same cache, then decode."""
    injected = format_result(decision, fmt) + follow_up
    return generate_from(clf, branch_cache, clf.encode(injected), max_tokens, temp)


def ab_injection(
    clf: Classifier,
    make_branch,
    decision: Decision,
    follow_up: str,
    formats: Sequence[str] = ("json", "prose", "bare"),
    max_tokens: int = 96,
) -> dict[str, GenResult]:
    """A/B the injection format on a fresh branch each time, so they are comparable."""
    out = {}
    for fmt in formats:
        cache = make_branch()
        out[fmt] = classify_then_generate(
            clf, cache, decision, follow_up, fmt=fmt, max_tokens=max_tokens
        )
        del cache
    return out
