"""Jev-style classification: one forward pass, no autoregressive decode.

The whole technique is three steps:
  1. prefill the shared state into a KV cache (done once, reused by every question)
  2. extend the cache with a short question, keeping only the *last* position's logits
  3. mask those logits to a fixed label-token set and renormalise

Step 3 is what makes this a classifier rather than a generator: the output
distribution is closed over a known label set, so there is nothing to parse and
nothing to hallucinate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

PREFILL_CHUNK = 512


class MultiTokenLabelError(ValueError):
    """A label surface form did not resolve to exactly one token."""


@dataclass
class LabelSet:
    """Fixed label set, resolved to single token ids.

    Single-token labels are a hard requirement: the decision is read from one
    position's logits, so a label spanning two tokens cannot be scored there
    without a decode step, which is the thing we are avoiding.
    """

    names: list[str]
    token_ids: list[int]
    surface_forms: list[str]

    @classmethod
    def resolve(
        cls,
        tokenizer: Any,
        labels: Sequence[str],
        variants: Sequence[str] = ("{}", " {}", "{capitalized}", " {capitalized}"),
    ) -> "LabelSet":
        token_ids: list[int] = []
        surfaces: list[str] = []
        for label in labels:
            chosen = None
            for pattern in variants:
                surface = pattern.format(label, capitalized=label.capitalize())
                ids = tokenizer.encode(surface, add_special_tokens=False)
                if len(ids) == 1:
                    chosen = (ids[0], surface)
                    break
            if chosen is None:
                raise MultiTokenLabelError(
                    f"label {label!r} has no single-token surface form; "
                    "pick a different label word (e.g. 'yes'/'no', 'A'/'B')"
                )
            token_ids.append(chosen[0])
            surfaces.append(chosen[1])
        if len(set(token_ids)) != len(token_ids):
            raise MultiTokenLabelError(
                f"labels collide on the same token: {list(zip(labels, token_ids))}"
            )
        return cls(names=list(labels), token_ids=token_ids, surface_forms=surfaces)

    def __len__(self) -> int:
        return len(self.names)


@dataclass
class Decision:
    label: str
    index: int
    probs: dict[str, float]
    margin: float           # top prob minus runner-up; the usable confidence signal
    entropy: float
    raw_logits: list[float]
    calibrated: bool = False
    meta: dict = field(default_factory=dict)

    def as_json(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.probs[self.label], 4),
            "margin": round(self.margin, 4),
            "probs": {k: round(v, 4) for k, v in self.probs.items()},
        }


def _decide(logits_row: mx.array, labels: LabelSet, bias: mx.array | None) -> Decision:
    """Mask full-vocab logits to the label set, then renormalise."""
    sel = logits_row[mx.array(labels.token_ids)].astype(mx.float32)
    raw = [float(x) for x in sel.tolist()]
    if bias is not None:
        sel = sel - bias
    probs = mx.softmax(sel)
    p = [float(x) for x in probs.tolist()]
    order = sorted(range(len(p)), key=lambda i: p[i], reverse=True)
    top = order[0]
    runner = p[order[1]] if len(order) > 1 else 0.0
    entropy = -sum(x * math.log(max(x, 1e-12)) for x in p)
    return Decision(
        label=labels.names[top],
        index=top,
        probs={labels.names[i]: p[i] for i in range(len(p))},
        margin=p[top] - runner,
        entropy=entropy,
        raw_logits=raw,
        calibrated=bias is not None,
    )


class Classifier:
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    # -- state -------------------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def prefill(self, tokens: Sequence[int], cache: list | None = None) -> list:
        """Prefill shared state. Chunked so a 6K-token state does not spike memory."""
        if cache is None:
            cache = make_prompt_cache(self.model)
        toks = list(tokens)
        if not toks:
            return cache          # empty state is legal: every question stands alone
        ids = mx.array(toks)[None]
        for i in range(0, ids.shape[1], PREFILL_CHUNK):
            chunk = ids[:, i : i + PREFILL_CHUNK]
            self.model(chunk, cache=cache)
            mx.eval([c.state for c in cache])
        return cache

    # -- the single forward pass -------------------------------------------

    def logits_at_end(self, cache: list, tokens: Sequence[int]) -> mx.array:
        """Extend `cache` with `tokens`; return next-token logits at the final position.

        Only the LAST position's logits are ever read, so everything before it
        is prefilled in chunks and only the final token is run alone. A single
        unchunked pass would materialise [1, n_tokens, vocab] -- at Qwen3.5's
        248k vocabulary a 3.7k-token prompt is ~3.7 GB of logits to read one
        row of, which is an OOM on long states and a large latency tail on
        merely-long ones.

        This mutates `cache` by design -- generation can continue straight from
        here. Use branch.py when you need the state left pristine.
        """
        toks = list(tokens)
        if not toks:
            raise ValueError("logits_at_end needs at least one token")
        head, last = toks[:-1], toks[-1:]
        for i in range(0, len(head), PREFILL_CHUNK):
            self.model(mx.array(head[i : i + PREFILL_CHUNK])[None], cache=cache)
            mx.eval([c.state for c in cache])
        row = self.model(mx.array(last)[None], cache=cache)[0, -1, :]
        mx.eval(row)
        return row

    def classify(
        self,
        cache: list,
        question_tokens: Sequence[int],
        labels: LabelSet,
        bias: mx.array | None = None,
    ) -> Decision:
        row = self.logits_at_end(cache, question_tokens)
        return _decide(row, labels, bias)

    # -- calibration -------------------------------------------------------

    def calibration_bias(
        self,
        state_tokens: Sequence[int],
        question_template: str,
        labels: LabelSet,
        content_free: Sequence[str] = ("N/A", "", "[MASK]"),
    ) -> mx.array:
        """Contextual calibration (Zhao et al. 2021).

        Ask the same question about content-free inputs; whatever label the model
        prefers there is prior bias, not signal. Subtract it in log space.
        """
        rows = []
        for cf in content_free:
            cache = self.prefill(state_tokens)
            q = self.encode(question_template.format(cf))
            row = self.logits_at_end(cache, q)
            sel = row[mx.array(labels.token_ids)].astype(mx.float32)
            rows.append(mx.log(mx.softmax(sel)))
            del cache
        return mx.mean(mx.stack(rows), axis=0)
