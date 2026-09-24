"""Head-duplication surgery on a frozen checkpoint -- no training.

Take Qwen2.5-0.5B's 14 query heads and duplicate them to 28. The duplicates
share W_q/W_k/W_v exactly, so on their own they would be redundant; two
perturbations make them do something different:

  * RoPE phase shift -- duplicates rotate queries at position p+shift, so they
    read the same content through a lookahead positional lens.
  * query-bias offset -- duplicates get q_proj.bias + delta. This is NOT a
    uniform shift of the attention scores (which would cancel in the softmax):
    adding d to the query adds d . k_j to each score, and k_j varies per key,
    so the attention distribution genuinely changes.

o_proj's columns are duplicated at HALF weight each, which makes the whole
transform the exact identity when shift=0 and delta=0. That is the control:
if a zero-perturbation franken model does not reproduce the original logits
bit-for-bit, the surgery is wired wrong and no other number can be trusted.

K and V are untouched, so the KV cache keeps its original shape and every
branching strategy still applies.

GQA makes the layout load-bearing. With 14 query heads over 2 KV heads, heads
0-6 read KV group 0 and 7-13 read group 1. Appending a second block of 14 would
renumber them so that heads 0-13 all read group 0 -- silently pairing every
query head with the wrong keys. Each duplicate therefore has to land INSIDE its
own group: every group goes from r heads to 2r, originals first, duplicates
second. The control below is what catches this.
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention


class DupHeadAttention(nn.Module):
    """Attention with query heads duplicated, phase-shifted and bias-offset."""

    def __init__(self, a: Any, delta: float = 0.2, shift: int = 1, halve: bool = True):
        super().__init__()
        self.n_orig = a.n_heads
        self.n_heads = a.n_heads * 2
        self.n_kv_heads = a.n_kv_heads
        self.scale = a.scale
        self.shift = shift
        self.rope = a.rope
        self.k_proj, self.v_proj = a.k_proj, a.v_proj

        self.r = r = a.n_heads // a.n_kv_heads      # query heads per KV group
        g, hd = a.n_kv_heads, a.q_proj.weight.shape[0] // a.n_heads
        self.head_dim = hd

        # duplicate WITHIN each KV group: (g, r, ...) -> (g, 2r, ...)
        W = a.q_proj.weight.reshape(g, r, hd, -1)
        self.q_w = mx.concatenate([W, W], axis=1).reshape(g * 2 * r * hd, -1)
        b = a.q_proj.bias.reshape(g, r, hd)
        self.q_b = mx.concatenate([b, b + delta], axis=1).reshape(-1)

        f = 0.5 if halve else 1.0
        Wo = (a.o_proj.weight * f).reshape(-1, g, r, hd)
        self.o_w = mx.concatenate([Wo, Wo], axis=2).reshape(-1, g * 2 * r * hd)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None,
                 cache: Optional[Any] = None) -> mx.array:
        B, L, _ = x.shape
        g, r, hd = self.n_kv_heads, self.r, self.head_dim
        q = x @ self.q_w.T + self.q_b
        k, v = self.k_proj(x), self.v_proj(x)

        # (B, L, g, 2r, hd) -> (B, g, 2r, L, hd); originals [:r], duplicates [r:]
        q = q.reshape(B, L, g, 2 * r, hd).transpose(0, 2, 3, 1, 4)
        k = k.reshape(B, L, g, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, g, hd).transpose(0, 2, 1, 3)

        off = cache.offset if cache is not None else 0
        qa = self.rope(q[:, :, :r].reshape(B, g * r, L, hd), offset=off)
        qb = self.rope(q[:, :, r:].reshape(B, g * r, L, hd), offset=off + self.shift)
        q = mx.concatenate([qa.reshape(B, g, r, L, hd),
                            qb.reshape(B, g, r, L, hd)], axis=2
                           ).reshape(B, g * 2 * r, L, hd)
        k = self.rope(k, offset=off)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = scaled_dot_product_attention(q, k, v, cache=cache,
                                           scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return out @ self.o_w.T


def frankenize(model: Any, delta: float = 0.2, shift: int = 1,
               layers: Optional[slice] = None) -> Any:
    """Replace each layer's attention in place. Returns the same model object."""
    blocks = model.model.layers
    sel = range(len(blocks)) if layers is None else range(*layers.indices(len(blocks)))
    for i in sel:
        blocks[i].self_attn = DupHeadAttention(blocks[i].self_attn, delta, shift)
    return model
