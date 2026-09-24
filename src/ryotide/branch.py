"""Parallel classification over one shared state, via branched KV caches.

Spec step 4. Deliberately NOT masked single-pass batched attention -- that is
deferred until branching is measured to be the bottleneck.

Three strategies, because they trade memory against the ability to continue
generating from a branch:

  fork  -- fork the cache per question, keep every branch alive.
           memory: N x prefix. Any branch can continue generating immediately.
  trim  -- one cache; after each question, trim the offset back to the end of
           the shared state and overwrite in place.
           memory: 1 x prefix. No branch survives; nothing can continue.
  lazy  -- classify with `trim`, then fork+replay only the branch you actually
           want to generate from. Replay costs one short question prefill.
           memory: 1 x prefix (+1 fork on demand). Default.

`lazy` is the default because a question is ~10-30 tokens: re-running one is
far cheaper than holding N full prefix copies resident the whole time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, RotatingKVCache, make_prompt_cache

from .classify import Classifier, Decision, LabelSet


def fork_cache(cache: list) -> list:
    """Copy-on-write fork of a prompt cache.

    Cheap at fork time: `state` hands back arrays without copying. The real
    copy lands on the branch's first extension, when update_and_fetch takes its
    concatenate path rather than writing into the parent's buffer. That is also
    what makes it safe -- a branch never writes into a sibling's memory.
    """
    forked = []
    for c in cache:
        # Hybrid models (Qwen3.5) mix linear-attention layers in among the
        # attention ones. Those keep a fixed-size recurrent state, not a
        # growing KV buffer: no offset, nothing to rewind. Their slots are
        # rebound on write rather than mutated, so a copied list is enough
        # to keep siblings independent -- verified empirically, not assumed.
        if isinstance(c, ArraysCache):
            new = ArraysCache(len(c.cache))
            new.cache = list(c.cache)
            forked.append(new)
            continue
        # Sliding-window layers (Gemma 4: 20 of 24) keep a ring buffer. Once it
        # has wrapped, `state` hands back the buffer itself and single-token
        # writes land IN PLACE -- a copy-on-write fork would then scribble over
        # its parent. Give each branch its own buffer. It is bounded by the
        # window (512 tokens), so the copy is cheap.
        if isinstance(c, RotatingKVCache):
            new = RotatingKVCache(max_size=c.max_size, keep=c.keep)
            if getattr(c, "keys", None) is not None:
                k, v = c.state
                new.state = (k * 1, v * 1)
                new.meta_state = c.meta_state
            forked.append(new)
            continue
        new = type(c)()
        if getattr(c, "keys", None) is None:
            forked.append(new)   # nothing prefilled yet; a fresh cache is the fork
            continue
        if hasattr(c, "meta_state"):
            try:
                new.meta_state = c.meta_state
            except Exception:
                pass
        new.state = c.state
        forked.append(new)
    return forked


def cache_bytes(cache: list) -> int:
    total = 0
    for c in cache:
        if getattr(c, "keys", None) is None:
            continue
        k, v = c.state
        total += k.nbytes + v.nbytes
    return total


def cache_offset(cache: list) -> int:
    return cache[0].offset


def supports_trim(cache: list) -> bool:
    """True only if every layer can actually rewind."""
    return all(getattr(c, "is_trimmable", lambda: False)() for c in cache)


def trim_to(cache: list, length: int) -> None:
    """Drop everything past `length`. O(1) -- just moves the write head back.

    Refuses on any cache that cannot rewind. A recurrent state has no offset to
    roll back, so a silent no-op here would leak one question into the next --
    exactly the cross-contamination this design exists to prevent.
    """
    if not supports_trim(cache):
        kinds = sorted({type(c).__name__ for c in cache
                        if not getattr(c, "is_trimmable", lambda: False)()})
        raise TypeError(
            f"cache is not trimmable ({', '.join(kinds)}): strategies 'trim' and "
            "'lazy' are unsound on this model -- use strategy='fork'")
    for c in cache:
        n = c.offset - length
        if n > 0:
            c.trim(n)


@dataclass
class BranchResult:
    question: str
    decision: Decision
    cache: list | None = None   # populated only under strategy="fork"


class BranchedClassifier:
    def __init__(self, clf: Classifier):
        self.clf = clf
        self.state_cache: list | None = None
        self.state_len: int = 0
        self.state_tokens: list[int] = []

    def set_state(self, text: str) -> int:
        """Prefill the shared state once. Every question reuses this."""
        self.state_tokens = self.clf.encode(text)
        self.state_cache = self.clf.prefill(self.state_tokens)
        self.state_len = cache_offset(self.state_cache)
        return self.state_len

    def classify_many(
        self,
        questions: Sequence[str],
        labels: LabelSet,
        strategy: str = "lazy",
        bias: mx.array | None = None,
    ) -> list[BranchResult]:
        if self.state_cache is None:
            raise RuntimeError("call set_state() first")
        if strategy not in ("fork", "trim", "lazy"):
            raise ValueError(f"unknown strategy {strategy!r}")

        results: list[BranchResult] = []
        for q in questions:
            q_tokens = self.clf.encode(q)
            if strategy == "fork":
                branch = fork_cache(self.state_cache)
                d = self.clf.classify(branch, q_tokens, labels, bias=bias)
                results.append(BranchResult(q, d, cache=branch))
            else:
                # Extend the shared cache, read the logit, rewind. Each question
                # sees only the state -- never a sibling's question or answer,
                # which is the cross-contamination property we are after.
                d = self.clf.classify(self.state_cache, q_tokens, labels, bias=bias)
                trim_to(self.state_cache, self.state_len)
                results.append(BranchResult(q, d, cache=None))
        return results

    def materialize(self, question: str) -> list:
        """Rebuild one branch's cache on demand (strategy='lazy' path)."""
        if self.state_cache is None:
            raise RuntimeError("call set_state() first")
        branch = fork_cache(self.state_cache)
        self.clf.logits_at_end(branch, self.clf.encode(question))
        return branch


def projected_branch_memory(kv_bytes_per_token: int, state_len: int, n: int) -> dict:
    """The arithmetic the spec flags as never having been done."""
    per_branch = kv_bytes_per_token * state_len
    return {
        "kv_bytes_per_token": kv_bytes_per_token,
        "state_len": state_len,
        "per_branch_mb": per_branch / 1e6,
        "n_branches": n,
        "fork_total_mb": per_branch * n / 1e6,
        "trim_total_mb": per_branch / 1e6,
        "lazy_total_mb": per_branch * 2 / 1e6,
    }
