"""PyTorch backend: the same single forward pass, on CUDA, Apple MPS or CPU.

The MLX path is the reference implementation; this one exists so RYOTIDE runs
where MLX does not (NVIDIA GPUs, Linux). Everything that decides WHAT is read --
prompt, chat template, answer prefix, option markers, label folding -- lives in
the adapter and is shared; this class only turns token ids into the masked
next-token distribution at the final position.

No KV-cache branching here: each read is one full forward pass. With one option
order (the default configuration) that is exactly what the MLX path does too.
"""
from __future__ import annotations

from typing import Sequence


def _pick_device(device: str | None) -> str:
    import torch
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TorchClassifier:
    def __init__(self, model_id: str, revision: str | None = None,
                 device: str | None = None, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = _pick_device(device)
        self.dtype = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=self.dtype).to(self.device).eval()

    def read(self, tokens: Sequence[int], marker_ids: Sequence[int]) -> tuple[list[float], float]:
        """One forward pass; the next-token distribution at the LAST position.

        Returns (probabilities over the markers, renormalised) and the share of
        the full-vocabulary softmax that sat on the markers before masking.
        """
        torch = self.torch
        ids = torch.tensor([list(tokens)], device=self.device)
        with torch.inference_mode():
            # logits_to_keep=1: never materialise [n_tokens, vocab] logits
            out = self.model(input_ids=ids, logits_to_keep=1, use_cache=False)
        row = out.logits[0, -1].float()
        sel = torch.tensor(list(marker_ids), device=self.device)
        mass = float(torch.softmax(row, -1)[sel].sum())
        probs = torch.softmax(row[sel], -1).tolist()
        return probs, mass
