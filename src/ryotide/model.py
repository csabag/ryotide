"""Model loading and introspection.

Raw mlx-lm only (spec build-order step 2): we need direct logit access at a
chosen sequence position, which the generation-oriented wrappers do not expose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lm.utils import load

DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


@dataclass
class ModelInfo:
    name: str
    n_layers: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    kv_dtype_bytes: int

    @property
    def kv_bytes_per_token(self) -> int:
        """K and V, every layer. This is the number that drives branch memory."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.kv_dtype_bytes


def load_model(name: str = DEFAULT_MODEL) -> tuple[Any, Any]:
    model, tokenizer = load(name)
    return model, tokenizer


def describe(model: Any, name: str = DEFAULT_MODEL) -> ModelInfo:
    args = model.args
    hidden = args.hidden_size
    n_heads = args.num_attention_heads
    head_dim = getattr(args, "head_dim", None) or hidden // n_heads
    n_kv_heads = getattr(args, "num_key_value_heads", n_heads)
    # KV cache is stored at compute dtype, not the quantized weight dtype.
    dtype = model.model.embed_tokens(mx.array([[0]])).dtype
    return ModelInfo(
        name=name,
        n_layers=args.num_hidden_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden,
        vocab_size=args.vocab_size,
        kv_dtype_bytes=dtype.size,
    )
